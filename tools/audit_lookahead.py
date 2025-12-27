#!/usr/bin/env python3
"""
================================================================================
AUDITORIA DE LOOK-AHEAD BIAS
================================================================================

Analisa o codigo do backtest para identificar vazamentos de informacao futura.
================================================================================
"""

print("=" * 80)
print("  AUDITORIA DE LOOK-AHEAD BIAS")
print("=" * 80)

print("""
================================================================================
ANALISE DO CODIGO: hmm_m5_high_frequency.py
================================================================================

FLUXO DO LOOP PRINCIPAL (linhas 279-379):

    for i in range(warmup_bars, total_bars):
        bar = bars[i]  # Barra ATUAL

        # PASSO 1: Executar sinal pendente (da barra anterior)
        if pending_signal is not None and position is None:
            entry_price = bar.ask_open  # Usa OPEN da barra atual - OK

        # PASSO 2: Verificar stop/take profit
        if position:
            if bid_low <= position.stop_loss:  # Usa LOW da barra atual
                exit_price = position.stop_loss

        # PASSO 3: Atualizar indicador
        indicator.update(bar)  # <-- USA bar.close (CLOSE DA BARRA ATUAL)

        # PASSO 4: Gerar sinal para proxima barra
        if signal_valid:
            trend = indicator.get_trend_direction(lookback)
            pending_signal = 'SELL' if trend == 1 else 'BUY'

================================================================================
PROBLEMAS IDENTIFICADOS:
================================================================================

PROBLEMA 1: USO DO CLOSE DA BARRA ATUAL (CRITICO)
--------------------------------------------------------------------------------
Linha 172: ret = np.log(bar.close / self.last_price)
Linha 185: self.closes_buffer.append(bar.close)

O indicador usa bar.close para calcular:
- Volatilidade (desvio padrao dos retornos)
- Direcao da tendencia (closes[-1] > closes[-lookback-1])

Em tempo real, o CLOSE da barra N so esta disponivel APOS a barra N terminar.
Mas o codigo usa esse close para gerar sinal NA MESMA barra N.

Exemplo de look-ahead:
- Barra N abre as 10:00, fecha as 10:05
- As 10:04, voce NAO sabe qual sera o close
- Mas o backtest usa o close para decidir na barra N

SEVERIDADE: ALTA
IMPACTO: O sinal usa informacao que nao estaria disponivel em tempo real


PROBLEMA 2: ORDEM DE EXECUCAO
--------------------------------------------------------------------------------
O loop faz:
1. Executa sinal pendente (OK - usa open da barra atual)
2. Verifica stop/take (usa high/low da barra atual)
3. Atualiza indicador com close da barra atual  <-- PROBLEMA
4. Gera sinal baseado no indicador atualizado   <-- USA CLOSE ATUAL

A ordem CORRETA seria:
1. Executa sinal pendente
2. Verifica stop/take
3. Gera sinal baseado em dados ATE BARRA ANTERIOR
4. Atualiza indicador (para proxima iteracao)

SEVERIDADE: ALTA


PROBLEMA 3: AMBIGUIDADE NO STOP/TAKE
--------------------------------------------------------------------------------
Linhas 318-329:
    if bid_low <= position.stop_loss:
        exit_price = position.stop_loss
    elif bid_high >= position.take_profit:
        exit_price = position.take_profit

Se AMBOS stop e take sao atingidos na mesma barra, qual foi primeiro?
O codigo assume que stop tem prioridade, mas isso pode nao ser verdade.

Exemplo:
- Entry: 1.1000, SL: 1.0990, TP: 1.1020
- Barra: Open=1.1000, High=1.1025, Low=1.0985, Close=1.1010
- Tanto SL quanto TP foram atingidos. Qual primeiro?

O backtest assume STOP primeiro, mas poderia ter sido TP primeiro.

SEVERIDADE: MEDIA
IMPACTO: Pode subestimar ou superestimar resultados


PROBLEMA 4: SLIPPAGE NAO CONSIDERADO
--------------------------------------------------------------------------------
Linhas 320-321:
    if bid_low <= position.stop_loss:
        exit_price = position.stop_loss  # Assume execucao EXATA no stop

Em mercado real, stops podem ter slippage (executar em preco pior).
O backtest assume execucao perfeita no preco do stop.

SEVERIDADE: BAIXA
IMPACTO: Resultados levemente otimistas


================================================================================
CORRECOES NECESSARIAS:
================================================================================

CORRECAO 1: Usar dados da barra ANTERIOR para gerar sinais
--------------------------------------------------------------------------------

ANTES (ERRADO):
    indicator.update(bar)  # Atualiza com close atual
    if signal_valid:
        trend = indicator.get_trend_direction(lookback)  # Usa close atual
        pending_signal = ...

DEPOIS (CORRETO):
    # Gerar sinal ANTES de atualizar o indicador
    if signal_valid:
        trend = indicator.get_trend_direction(lookback)  # Usa ate close anterior
        pending_signal = ...

    indicator.update(bar)  # Atualiza DEPOIS para proxima iteracao


CORRECAO 2: Mover update() para o final do loop
--------------------------------------------------------------------------------

ORDEM CORRETA:
    for i in range(warmup_bars, total_bars):
        bar = bars[i]

        # 1. Executar sinal pendente (usa OPEN da barra atual - OK)
        if pending_signal and position is None:
            execute_entry(bar.open)

        # 2. Verificar stop/take (usa HIGH/LOW da barra atual - OK)
        if position:
            check_exit(bar.high, bar.low)

        # 3. Gerar sinal (USA APENAS DADOS ATE BARRA ANTERIOR)
        if position is None and pending_signal is None:
            signal = generate_signal()  # Indicador ainda tem dados ate N-1
            pending_signal = signal

        # 4. Atualizar indicador COM barra atual (para PROXIMA iteracao)
        indicator.update(bar)


CORRECAO 3: Usar bar.open para trend direction
--------------------------------------------------------------------------------

Alternativa: usar open da barra atual em vez de close:

    def get_trend_direction_safe(self, current_open: float, lookback: int):
        closes = list(self.closes_buffer)
        # Compara OPEN atual com close de lookback barras atras
        if current_open > closes[-lookback]:
            return 1
        else:
            return -1

Isso garante que usamos apenas informacao disponivel no inicio da barra.


================================================================================
RESUMO DA AUDITORIA:
================================================================================

PROBLEMAS ENCONTRADOS:
  [CRITICO] Uso do close da barra atual para gerar sinal
  [CRITICO] Ordem de execucao incorreta (update antes de gerar sinal)
  [MEDIO]   Ambiguidade quando stop e take sao atingidos na mesma barra
  [BAIXO]   Slippage nao considerado

RECOMENDACAO:
  Reordenar o loop para gerar sinal ANTES de atualizar indicador.
  Isso garante que o sinal usa apenas dados ate a barra anterior.

================================================================================
""")

print("=" * 80)
print("  FIM DA AUDITORIA")
print("=" * 80)
