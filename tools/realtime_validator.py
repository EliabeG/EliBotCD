#!/usr/bin/env python3
"""
================================================================================
REALTIME BOT VALIDATOR
Monitoramento contínuo do EliBotCD em tempo real
================================================================================

Este script:
1. Conecta ao feed de dados em tempo real
2. Monitora todas as 30 estratégias
3. Valida sinais gerados
4. Registra performance em tempo real
5. Detecta anomalias e problemas

Uso:
    python tools/realtime_validator.py [--duration MINUTES] [--verbose]
"""

import asyncio
import sys
import os
import signal
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
import argparse

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from api.fxopen_client import FXOpenClient, Tick
from strategies.orchestrator import StrategyOrchestrator
from strategies.strategy_factory import create_all_strategies, get_strategy_count


@dataclass
class IndicatorStats:
    """Estatísticas de um indicador"""
    name: str
    signals_generated: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    last_signal_time: Optional[datetime] = None
    avg_confidence: float = 0.0
    confidences: List[float] = field(default_factory=list)
    errors: int = 0
    last_error: Optional[str] = None


@dataclass
class ValidationResult:
    """Resultado de validação"""
    timestamp: datetime
    check_name: str
    passed: bool
    message: str
    severity: str = "INFO"  # INFO, WARNING, ERROR, CRITICAL


class RealtimeValidator:
    """
    Validador em tempo real do EliBotCD
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.running = False

        # Cliente API
        self.client = FXOpenClient()

        # Orquestrador
        self.orchestrator = StrategyOrchestrator(
            symbol=settings.SYMBOL,
            min_confidence=0.3,  # Baixa para capturar mais sinais
            signal_cooldown_ticks=50
        )

        # Estatísticas
        self.start_time: Optional[datetime] = None
        self.tick_count = 0
        self.last_tick_time: Optional[datetime] = None
        self.tick_gaps: deque = deque(maxlen=1000)

        # Indicadores
        self.indicator_stats: Dict[str, IndicatorStats] = {}

        # Volatilidade
        self.volatility_history: deque = deque(maxlen=1000)
        self.volatility_changes: List[Dict] = []

        # Sinais
        self.signals_log: List[Dict] = []
        self.signal_count = 0

        # Validações
        self.validations: List[ValidationResult] = []

        # Preços
        self.price_history: deque = deque(maxlen=1000)

        # Registra estratégias
        self._register_strategies()

    def _register_strategies(self):
        """Registra todas as estratégias"""
        print("\n" + "=" * 70)
        print("  REGISTRANDO ESTRATÉGIAS")
        print("=" * 70)

        strategies = create_all_strategies()

        for level, strats in strategies.items():
            print(f"\n{level} VOLATILIDADE ({len(strats)} estratégias):")
            for strategy in strats:
                self.orchestrator.register_strategy(strategy, level)
                self.indicator_stats[strategy.name] = IndicatorStats(name=strategy.name)
                if self.verbose:
                    print(f"  - {strategy.name}")

        counts = get_strategy_count()
        print(f"\nTotal: {counts['TOTAL']} estratégias registradas")
        print("=" * 70)

    async def on_tick(self, tick: Tick):
        """Callback para cada tick"""
        now = datetime.now(timezone.utc)
        self.tick_count += 1

        # Calcula gap entre ticks
        if self.last_tick_time:
            gap = (now - self.last_tick_time).total_seconds()
            self.tick_gaps.append(gap)

            # Valida gap
            if gap > 30:
                self._add_validation(
                    "TICK_GAP",
                    False,
                    f"Gap de {gap:.1f}s entre ticks detectado",
                    "WARNING"
                )

        self.last_tick_time = now

        # Armazena preço
        self.price_history.append({
            'price': tick.mid,
            'timestamp': tick.timestamp,
            'spread': tick.spread
        })

        # Processa no orquestrador
        try:
            signal = self.orchestrator.process_tick(
                price=tick.mid,
                timestamp=tick.timestamp,
                volume=tick.bid_volume + tick.ask_volume
            )

            # Atualiza estado de volatilidade
            state = self.orchestrator.get_state()
            self.volatility_history.append({
                'volatility': state.current_volatility,
                'level': state.volatility_level,
                'hurst': state.hurst,
                'entropy': state.entropy,
                'timestamp': now
            })

            # Detecta mudança de volatilidade
            if len(self.volatility_history) >= 2:
                prev = self.volatility_history[-2]
                curr = self.volatility_history[-1]
                if prev['level'] != curr['level']:
                    self.volatility_changes.append({
                        'from': prev['level'],
                        'to': curr['level'],
                        'timestamp': now,
                        'volatility': curr['volatility']
                    })
                    self._print_volatility_change(prev['level'], curr['level'])

            # Processa sinal se houver
            if signal:
                self._process_signal(signal, state)

            # Exibe status periódico
            if self.tick_count % 100 == 0:
                self._print_status(tick, state)

        except Exception as e:
            self._add_validation(
                "TICK_PROCESSING",
                False,
                f"Erro ao processar tick: {e}",
                "ERROR"
            )

    def _process_signal(self, signal, state):
        """Processa um sinal gerado"""
        self.signal_count += 1

        # Registra sinal
        signal_data = {
            'number': self.signal_count,
            'type': signal.type.name,
            'price': signal.price,
            'confidence': signal.confidence,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'strategy': signal.strategy_name,
            'reason': signal.reason,
            'timestamp': signal.timestamp.isoformat(),
            'volatility_level': state.volatility_level,
            'volatility': state.current_volatility,
            'hurst': state.hurst,
            'entropy': state.entropy
        }
        self.signals_log.append(signal_data)

        # Atualiza stats do indicador
        # Extrai nomes das estratégias do reason
        if 'Agregado' in signal.reason:
            # Sinal agregado
            pass
        else:
            # Sinal individual
            if signal.strategy_name in self.indicator_stats:
                stats = self.indicator_stats[signal.strategy_name]
                stats.signals_generated += 1
                if signal.type.name == 'BUY':
                    stats.buy_signals += 1
                else:
                    stats.sell_signals += 1
                stats.last_signal_time = signal.timestamp
                stats.confidences.append(signal.confidence)
                stats.avg_confidence = sum(stats.confidences) / len(stats.confidences)

        # Exibe sinal
        self._print_signal(signal, state)

        # Valida sinal
        self._validate_signal(signal, state)

    def _validate_signal(self, signal, state):
        """Valida um sinal"""
        # Verifica stop loss
        if signal.stop_loss:
            sl_distance = abs(signal.price - signal.stop_loss) * 10000
            if sl_distance < 5:
                self._add_validation(
                    "STOP_LOSS",
                    False,
                    f"Stop loss muito próximo: {sl_distance:.1f} pips",
                    "WARNING"
                )
            elif sl_distance > 100:
                self._add_validation(
                    "STOP_LOSS",
                    False,
                    f"Stop loss muito distante: {sl_distance:.1f} pips",
                    "WARNING"
                )
        else:
            self._add_validation(
                "STOP_LOSS",
                False,
                "Sinal sem stop loss definido",
                "ERROR"
            )

        # Verifica take profit
        if signal.take_profit:
            tp_distance = abs(signal.price - signal.take_profit) * 10000
            if tp_distance < 5:
                self._add_validation(
                    "TAKE_PROFIT",
                    False,
                    f"Take profit muito próximo: {tp_distance:.1f} pips",
                    "WARNING"
                )
        else:
            self._add_validation(
                "TAKE_PROFIT",
                False,
                "Sinal sem take profit definido",
                "ERROR"
            )

        # Verifica confiança
        if signal.confidence < 0.4:
            self._add_validation(
                "CONFIDENCE",
                False,
                f"Confiança baixa: {signal.confidence:.1%}",
                "INFO"
            )
        else:
            self._add_validation(
                "CONFIDENCE",
                True,
                f"Confiança adequada: {signal.confidence:.1%}",
                "INFO"
            )

    def _add_validation(self, check_name: str, passed: bool, message: str, severity: str):
        """Adiciona resultado de validação"""
        result = ValidationResult(
            timestamp=datetime.now(timezone.utc),
            check_name=check_name,
            passed=passed,
            message=message,
            severity=severity
        )
        self.validations.append(result)

        if severity in ("ERROR", "CRITICAL") or (not passed and severity == "WARNING"):
            color = "\033[91m" if severity in ("ERROR", "CRITICAL") else "\033[93m"
            reset = "\033[0m"
            print(f"\n{color}[{severity}] {check_name}: {message}{reset}")

    def _print_status(self, tick: Tick, state):
        """Exibe status periódico"""
        elapsed = datetime.now(timezone.utc) - self.start_time

        # Calcula tick rate
        avg_gap = sum(self.tick_gaps) / len(self.tick_gaps) if self.tick_gaps else 0
        tick_rate = 1 / avg_gap if avg_gap > 0 else 0

        # Cores para volatilidade
        colors = {
            'BAIXA': '\033[92m',
            'MEDIA': '\033[93m',
            'ALTA': '\033[91m',
            'INDEFINIDO': '\033[0m'
        }
        color = colors.get(state.volatility_level, '\033[0m')
        reset = '\033[0m'

        print(f"\n[{elapsed}] Tick #{self.tick_count}")
        print(f"  Preço: {tick.mid:.5f} | Spread: {tick.spread*10000:.1f} pips")
        print(f"  {color}Volatilidade: {state.current_volatility:.3f} pips | Nível: {state.volatility_level}{reset}")
        print(f"  Hurst: {state.hurst:.3f} | Entropy: {state.entropy:.3f}")
        print(f"  Sinais: {self.signal_count} | Tick Rate: {tick_rate:.1f}/s")
        print(f"  Estratégias ativas: {state.active_strategies}")

    def _print_signal(self, signal, state):
        """Exibe sinal com formatação"""
        color = '\033[92m' if signal.type.name == 'BUY' else '\033[91m'
        reset = '\033[0m'

        print(f"\n{'='*70}")
        print(f"{color}[SINAL #{self.signal_count}] {signal.type.name}{reset}")
        print(f"  Preço: {signal.price:.5f}")
        print(f"  Confiança: {signal.confidence:.1%}")
        print(f"  Stop Loss: {signal.stop_loss:.5f}" if signal.stop_loss else "  Stop Loss: N/A")
        print(f"  Take Profit: {signal.take_profit:.5f}" if signal.take_profit else "  Take Profit: N/A")
        print(f"  Estratégia: {signal.strategy_name}")
        print(f"  Volatilidade: {state.volatility_level} ({state.current_volatility:.3f} pips)")
        print(f"  Razão: {signal.reason[:100]}...")
        print(f"{'='*70}\n")

    def _print_volatility_change(self, from_level: str, to_level: str):
        """Exibe mudança de volatilidade"""
        colors = {
            'BAIXA': '\033[92m',
            'MEDIA': '\033[93m',
            'ALTA': '\033[91m'
        }

        from_color = colors.get(from_level, '\033[0m')
        to_color = colors.get(to_level, '\033[0m')
        reset = '\033[0m'

        print(f"\n{'*'*70}")
        print(f"  MUDANÇA DE VOLATILIDADE: {from_color}{from_level}{reset} → {to_color}{to_level}{reset}")
        print(f"{'*'*70}\n")

    def _print_final_report(self):
        """Exibe relatório final"""
        if not self.start_time:
            return

        elapsed = datetime.now(timezone.utc) - self.start_time

        print("\n" + "=" * 70)
        print("  RELATÓRIO FINAL DE VALIDAÇÃO")
        print("=" * 70)

        # Estatísticas gerais
        print(f"\n[RESUMO GERAL]")
        print(f"  Duração: {elapsed}")
        print(f"  Ticks processados: {self.tick_count}")
        print(f"  Sinais gerados: {self.signal_count}")
        print(f"  Taxa de sinais: {self.signal_count / (elapsed.total_seconds() / 60):.2f}/min" if elapsed.total_seconds() > 0 else "  Taxa de sinais: N/A")

        # Volatilidade
        print(f"\n[VOLATILIDADE]")
        print(f"  Mudanças de nível: {len(self.volatility_changes)}")
        if self.volatility_history:
            vols = [v['volatility'] for v in self.volatility_history]
            print(f"  Mín: {min(vols):.3f} pips | Máx: {max(vols):.3f} pips | Média: {sum(vols)/len(vols):.3f} pips")

        # Sinais por tipo
        print(f"\n[SINAIS POR TIPO]")
        buys = sum(1 for s in self.signals_log if s['type'] == 'BUY')
        sells = sum(1 for s in self.signals_log if s['type'] == 'SELL')
        print(f"  BUY: {buys} | SELL: {sells}")

        # Sinais por nível de volatilidade
        print(f"\n[SINAIS POR NÍVEL DE VOLATILIDADE]")
        by_level = defaultdict(int)
        for s in self.signals_log:
            by_level[s['volatility_level']] += 1
        for level in ['ALTA', 'MEDIA', 'BAIXA']:
            print(f"  {level}: {by_level[level]}")

        # Indicadores ativos
        print(f"\n[INDICADORES COM SINAIS]")
        active_indicators = [(name, stats) for name, stats in self.indicator_stats.items()
                            if stats.signals_generated > 0]
        active_indicators.sort(key=lambda x: x[1].signals_generated, reverse=True)

        if active_indicators:
            for name, stats in active_indicators[:10]:
                print(f"  {name}: {stats.signals_generated} sinais | "
                      f"BUY: {stats.buy_signals} | SELL: {stats.sell_signals} | "
                      f"Conf: {stats.avg_confidence:.1%}")
        else:
            print("  Nenhum indicador gerou sinais individuais")

        # Validações
        print(f"\n[VALIDAÇÕES]")
        errors = [v for v in self.validations if v.severity == 'ERROR']
        warnings = [v for v in self.validations if v.severity == 'WARNING']
        print(f"  Erros: {len(errors)}")
        print(f"  Avisos: {len(warnings)}")

        if errors:
            print(f"\n  Erros recentes:")
            for v in errors[-5:]:
                print(f"    - [{v.check_name}] {v.message}")

        # Conexão
        print(f"\n[CONEXÃO]")
        if self.tick_gaps:
            avg_gap = sum(self.tick_gaps) / len(self.tick_gaps)
            max_gap = max(self.tick_gaps)
            print(f"  Gap médio entre ticks: {avg_gap*1000:.0f}ms")
            print(f"  Maior gap: {max_gap:.1f}s")

        print("\n" + "=" * 70)

        # Salva log
        self._save_log()

    def _save_log(self):
        """Salva log de sinais em arquivo"""
        if not self.signals_log:
            return

        log_file = f"/home/azureuser/EliBotCD/logs/validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        log_data = {
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': datetime.now(timezone.utc).isoformat(),
            'tick_count': self.tick_count,
            'signal_count': self.signal_count,
            'volatility_changes': len(self.volatility_changes),
            'signals': self.signals_log,
            'validations_summary': {
                'errors': len([v for v in self.validations if v.severity == 'ERROR']),
                'warnings': len([v for v in self.validations if v.severity == 'WARNING'])
            }
        }

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)

        print(f"\nLog salvo em: {log_file}")

    async def run(self, duration_minutes: int = 0):
        """Executa o validador"""
        self.running = True
        self.start_time = datetime.now(timezone.utc)

        print("\n" + "=" * 70)
        print("  REALTIME BOT VALIDATOR - INICIANDO")
        print("=" * 70)
        print(f"  Símbolo: {settings.SYMBOL}")
        print(f"  Duração: {duration_minutes} min" if duration_minutes > 0 else "  Duração: Infinita")
        print(f"  Modo: VALIDAÇÃO (sem execução de ordens)")
        print("=" * 70 + "\n")

        # Conecta callback
        self.client.on_tick = self.on_tick

        try:
            # Conecta ao WebSocket
            if await self.client.connect():
                await self.client.subscribe(settings.SYMBOL)

                print(f"\nRecebendo ticks de {settings.SYMBOL}...")
                print("Pressione Ctrl+C para parar\n")
                print("-" * 70)

                # Loop principal
                if duration_minutes > 0:
                    await asyncio.sleep(duration_minutes * 60)
                else:
                    while self.running:
                        await asyncio.sleep(1)
            else:
                print("Falha ao conectar!")

        except KeyboardInterrupt:
            print("\n\nInterrompido pelo usuário")
        except Exception as e:
            print(f"\nErro: {e}")
        finally:
            self.running = False
            await self.client.disconnect()
            self._print_final_report()


async def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Validador em tempo real do EliBotCD"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Duração em minutos (0 = infinito)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Modo verbose"
    )

    args = parser.parse_args()

    validator = RealtimeValidator(verbose=args.verbose)
    await validator.run(duration_minutes=args.duration)


if __name__ == "__main__":
    asyncio.run(main())
