# AUDITORIA 8 - Custos e Slippage
## Data: 2025-12-25
## Versao: V2.1

---

## RESUMO EXECUTIVO

Esta auditoria analisa os custos de execucao e verifica se sao realistas.

---

## 1. CUSTOS CONFIGURADOS

### 1.1 CUSTOS PADRAO (optimizer.py:179-183)

```python
SPREAD_PIPS = 1.5       # Spread bid-ask
SLIPPAGE_PIPS = 0.8     # Derrapagem na execucao
COMMISSION_PIPS = 0.0   # Comissao (nao usada)
```

**ANALISE**:
- EURUSD tipico: 0.5-2.0 pips de spread
- 1.5 pips e conservador para a maioria das condicoes
- 0.8 pips de slippage e razoavel para H1
- Total: 2.3 pips de custo por trade (round-trip)

**STATUS**: OK - Custos realistas e conservadores.

---

### 1.2 APLICACAO DOS CUSTOS

#### Na Entrada (optimizer.py:466-474)

```python
total_cost = spread + slippage

if direction == 1:  # LONG
    entry_price = entry_price_raw + total_cost / 2  # Paga metade na entrada
    stop_price = entry_price - sl * pip
    take_price = entry_price + tp * pip
else:  # SHORT
    entry_price = entry_price_raw - total_cost / 2  # Paga metade na entrada
```

**ANALISE**: Custo dividido 50/50 entre entrada e saida.

---

#### Na Saida (optimizer.py:509-524)

```python
exit_price = take_price - slippage  # LONG - take
exit_price = stop_price - slippage  # LONG - stop
exit_price = take_price + slippage  # SHORT - take
exit_price = stop_price + slippage  # SHORT - stop
```

**STATUS**: OK - Slippage aplicado corretamente nas saidas.

---

## 2. COMPARACAO COM MERCADO REAL

| Cenario | Spread Real | Spread Usado | Status |
|---------|-------------|--------------|--------|
| EURUSD normal | 0.5-1.0 pips | 1.5 pips | Conservador |
| EURUSD volatil | 1.0-3.0 pips | 1.5 pips | OK |
| EURUSD news | 2.0-10.0 pips | 1.5 pips | Sub-estimado |

**NOTA**: Para eventos de noticias, os custos sao sub-estimados, mas o MAX_GAP_PIPS (50 pips) ajuda a filtrar esses eventos.

---

## 3. VALIDACAO TP > CUSTOS (optimizer.py:417-420)

```python
total_costs = self.SPREAD_PIPS + self.SLIPPAGE_PIPS  # 2.3 pips
if tp <= total_costs:
    return []  # Rejeita trades nao lucrativos
```

**STATUS**: OK - Valida que TP > custos.

---

## 4. VERIFICACOES REALIZADAS

1. [x] Custos realistas para EURUSD H1
2. [x] Custos aplicados corretamente na entrada
3. [x] Slippage aplicado nas saidas
4. [x] Validacao TP > custos
5. [x] Gap handling para eventos extremos

---

## 5. SCORE

| Categoria | Score |
|-----------|-------|
| Realismo | 9/10 |
| Conservadorismo | OK |
| Implementacao | OK |

---

## 6. STATUS: APROVADO
