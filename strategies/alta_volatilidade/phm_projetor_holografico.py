"""
Projetor Holográfico de Maldacena (PHM)
========================================
Nível de Complexidade: Física de Altas Energias / Teoria da Informação Quântica.

Premissa Teórica: Utilizaremos Redes Tensoriais (Tensor Networks), especificamente o
MERA (Multi-scale Entanglement Renormalization Ansatz), para construir a dimensão extra do
mercado. O preço é a borda (CFT). A rede tensorial é a geometria do Bulk (AdS). A volatilidade
extrema ocorre quando a Entropia de Entrelaçamento (Entanglement Entropy) entre
diferentes setores do mercado satura a capacidade de informação do espaço-tempo, criando
um "Buraco Negro" no Bulk.

Dependências Críticas: numpy (linalg.svd), scipy
"""

import numpy as np
from scipy.linalg import svd, expm
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')


class SpinChainEncoder:
    """
    Módulo 1: Tensorização do Estado do Mercado (Ψ)

    Transforme a janela de preços em um estado quântico de muitos corpos (Spin Chain).
    - Mapeie: Preço sobe = Spin Up |↑⟩, Preço desce = Spin Down |↓⟩.
    - Construa um Matrix Product State (MPS) inicial.
    """

    def __init__(self, physical_dim: int = 2):
        self.physical_dim = physical_dim
        self.eps = 1e-10
        self.spin_up = np.array([1.0, 0.0])
        self.spin_down = np.array([0.0, 1.0])

    def prices_to_spin_chain(self, prices: np.ndarray) -> np.ndarray:
        """Converte série de preços em cadeia de spins."""
        returns = np.diff(prices)
        n = len(returns)

        max_ret = np.max(np.abs(returns)) + self.eps
        amplitudes = returns / max_ret

        spin_states = np.zeros((n, 2))

        for i, amp in enumerate(amplitudes):
            if amp >= 0:
                theta = (np.pi / 4) * (1 - amp)
                spin_states[i] = np.array([np.cos(theta), np.sin(theta)])
            else:
                theta = (np.pi / 4) * (1 + amp)
                spin_states[i] = np.array([np.sin(theta), np.cos(theta)])

        return spin_states

    def create_mps(self, spin_states: np.ndarray, bond_dim: int = 4) -> list:
        """Constrói Matrix Product State (MPS) a partir da cadeia de spins."""
        n_sites = len(spin_states)
        mps_tensors = []

        for i in range(n_sites):
            if i == 0:
                tensor = np.zeros((1, self.physical_dim, bond_dim))
                for s in range(self.physical_dim):
                    tensor[0, s, s % bond_dim] = spin_states[i, s]
            elif i == n_sites - 1:
                tensor = np.zeros((bond_dim, self.physical_dim, 1))
                for s in range(self.physical_dim):
                    tensor[s % bond_dim, s, 0] = spin_states[i, s]
            else:
                tensor = np.zeros((bond_dim, self.physical_dim, bond_dim))
                for s in range(self.physical_dim):
                    tensor[s % bond_dim, s, s % bond_dim] = spin_states[i, s]

            mps_tensors.append(tensor)

        return mps_tensors


class SimplifiedMERA:
    """
    Módulo 2: Renormalização do Espaço (Construindo o Bulk) - MERA Simplificado

    Implementação eficiente do MERA usando SVD para coarse-graining.

    A Exigência Suprema: Truncamento de SVD Dinâmico. Descartar valores singulares
    menores que ε = 10^-5 para manter o Bond Dimension controlável.
    """

    def __init__(self, bond_dim: int = 8, n_layers: int = 4,
                 svd_cutoff: float = 1e-5, max_bond_dim: int = 16):
        self.bond_dim = bond_dim
        self.n_layers = n_layers
        self.svd_cutoff = svd_cutoff
        self.max_bond_dim = max_bond_dim
        self.eps = 1e-10
        self.entanglement_spectra = []

    def _truncated_svd(self, matrix: np.ndarray) -> tuple:
        """Truncamento de SVD Dinâmico com cutoff ε = 10^-5"""
        try:
            U, S, Vh = svd(matrix, full_matrices=False)
        except:
            return matrix[:, :1], np.array([1.0]), matrix[:1, :]

        S_normalized = S / (S[0] + self.eps)
        keep_mask = S_normalized > self.svd_cutoff
        n_keep = max(1, min(np.sum(keep_mask), self.max_bond_dim))

        return U[:, :n_keep], S[:n_keep], Vh[:n_keep, :]

    def coarse_grain_layer(self, mps_tensors: list) -> list:
        """Aplica coarse-graining para reduzir número de sites pela metade."""
        n = len(mps_tensors)
        if n < 2:
            return mps_tensors

        new_tensors = []

        for i in range(0, n - 1, 2):
            t1, t2 = mps_tensors[i], mps_tensors[i + 1]

            # Ajustar dimensões
            chi_m = min(t1.shape[2], t2.shape[0])
            t1 = t1[:, :, :chi_m]
            t2 = t2[:chi_m, :, :]

            # Contrair tensores
            contracted = np.tensordot(t1, t2, axes=([2], [0]))

            # Reshape para matriz
            chi_l, d1, d2, chi_r = contracted.shape
            matrix = contracted.reshape(chi_l * d1, d2 * chi_r)

            # SVD truncado
            U, S, Vh = self._truncated_svd(matrix)
            self.entanglement_spectra.append(S.copy())

            # Reconstruir tensor
            new_chi = len(S)
            new_tensor = (U @ np.diag(S)).reshape(chi_l, d1, new_chi)
            new_tensors.append(new_tensor)

        # Último tensor se ímpar
        if n % 2 == 1:
            new_tensors.append(mps_tensors[-1])

        return new_tensors

    def build_mera(self, mps_tensors: list) -> dict:
        """Constrói rede MERA completa (todas as camadas)."""
        self.entanglement_spectra = []
        layers = [mps_tensors]
        current = mps_tensors

        for layer_idx in range(self.n_layers):
            if len(current) < 2:
                break
            current = self.coarse_grain_layer(current)
            layers.append(current)

        return {
            'layers': layers,
            'n_layers': len(layers),
            'top_tensor': current[0] if current else None,
            'entanglement_spectra': self.entanglement_spectra
        }


class RyuTakayanagi:
    """
    Módulo 3: A Fórmula de Ryu-Takayanagi (Entropia Holográfica)

    S_A = Area(γ_A) / 4G_N

    - S_A baixa: Mercado "desconectado". Baixa correlação quântica.
    - S_A alta e estável: Tendência saudável.
    - Pico Súbito em S_A: Buraco Negro formando. Precede crash ou pump violento.
    """

    def __init__(self, G_N: float = 1.0):
        self.G_N = G_N
        self.eps = 1e-10

    def entropy_from_spectrum(self, singular_values: np.ndarray) -> float:
        """S = -Σ λ² log(λ²)"""
        if len(singular_values) == 0:
            return 0.0
        S = singular_values / (np.sum(singular_values) + self.eps)
        S2 = S ** 2
        return -np.sum(S2 * np.log(S2 + self.eps))

    def compute_entropy_profile(self, mps_tensors: list) -> np.ndarray:
        """Calcula perfil de entropia para diferentes cortes."""
        n = len(mps_tensors)
        if n < 2:
            return np.array([0.0])

        entropies = []

        # Calcular entropia em cada posição de corte
        for cut_pos in range(1, min(n, 20)):  # Limitar para eficiência
            # Contrair parte esquerda
            left = mps_tensors[0]
            for i in range(1, cut_pos):
                chi_m = min(left.shape[2], mps_tensors[i].shape[0])
                left = np.tensordot(left[:, :, :chi_m],
                                   mps_tensors[i][:chi_m, :, :],
                                   axes=([2], [0]))
                # Reshape se necessário
                if len(left.shape) == 4:
                    s = left.shape
                    left = left.reshape(s[0], s[1]*s[2], s[3])

            # SVD na interface
            try:
                matrix = left.reshape(-1, left.shape[-1])
                _, S, _ = svd(matrix, full_matrices=False)
                entropy = self.entropy_from_spectrum(S)
            except:
                entropy = 0.0

            entropies.append(entropy)

        return np.array(entropies)

    def detect_spike(self, entropy_series: np.ndarray,
                     percentile: float = 90) -> dict:
        """Detecta picos súbitos de entropia (formação de horizonte)."""
        if len(entropy_series) < 3:
            return {'spike_detected': False, 'magnitude': 0.0}

        d_entropy = np.gradient(entropy_series)
        threshold = np.percentile(np.abs(d_entropy), percentile)
        current = np.abs(d_entropy[-1]) if len(d_entropy) > 0 else 0

        return {
            'spike_detected': current > threshold,
            'magnitude': current / (threshold + self.eps),
            'threshold': threshold
        }


class HolographicComplexity:
    """
    Módulo 4: O Cálculo da Complexidade Holográfica

    Se o volume do Bulk cresce enquanto entropia estagna = "Stress Computacional"

    Sinal de Disparo:
    - dC/dt inverte o sinal drasticamente
    - Entropia cruza limiar crítico
    """

    def __init__(self):
        self.eps = 1e-10

    def compute_complexity(self, mera_result: dict) -> float:
        """Complexidade ∝ número de tensores × bond dimension."""
        total = 0.0
        for layer_idx, layer in enumerate(mera_result['layers']):
            depth = layer_idx + 1
            for tensor in layer:
                volume = np.prod(tensor.shape)
                total += volume * depth
        return total

    def from_spectra(self, spectra: list) -> float:
        """Complexidade dos espectros (rank efetivo)."""
        if not spectra:
            return 0.0

        complexity = 0.0
        for spec in spectra:
            if len(spec) > 0:
                S_norm = spec / (np.sum(spec) + self.eps)
                eff_rank = np.exp(-np.sum(S_norm * np.log(S_norm + self.eps)))
                complexity += eff_rank
        return complexity

    def detect_stress(self, complexity_series: np.ndarray,
                      entropy_series: np.ndarray) -> dict:
        """Detecta stress computacional."""
        if len(complexity_series) < 3 or len(entropy_series) < 3:
            return {'stress_detected': False, 'level': 0.0}

        c_trend = np.polyfit(range(len(complexity_series)), complexity_series, 1)[0]
        e_trend = np.polyfit(range(len(entropy_series)), entropy_series, 1)[0]

        stress_level = c_trend / (np.abs(e_trend) + self.eps)

        return {
            'stress_detected': stress_level > 2.0 and e_trend < c_trend * 0.1,
            'level': stress_level
        }


class IsingPhase:
    """
    Hamiltoniano de Ising para determinação de fase

    Ferromagnético → Tendência ordenada
    Paramagnético → Desordem/Ruído
    """

    def __init__(self):
        self.eps = 1e-10

    def magnetization(self, spin_states: np.ndarray) -> float:
        """⟨M⟩ = Σ⟨σ_i^z⟩/N"""
        n = len(spin_states)
        mag = 0.0
        for state in spin_states:
            mag += state[0]**2 - state[1]**2
        return mag / n

    def correlation(self, spin_states: np.ndarray, r: int = 1) -> float:
        """⟨σ_i^z σ_{i+r}^z⟩"""
        n = len(spin_states)
        if n <= r:
            return 0.0

        corr = 0.0
        for i in range(n - r):
            exp_i = spin_states[i, 0]**2 - spin_states[i, 1]**2
            exp_j = spin_states[i + r, 0]**2 - spin_states[i + r, 1]**2
            corr += exp_i * exp_j

        return corr / (n - r)

    def determine_phase(self, magnetization: float,
                        correlation: float) -> dict:
        """Determina fase do sistema."""
        abs_mag = np.abs(magnetization)

        # Thresholds ajustados para dados reais (mag tipica: -0.07 a 0.08)
        if abs_mag > 0.025:
            phase = "FERROMAGNETICO"
            desc = "Ordem Ferromagnética - Tendência"
        elif abs_mag < 0.01:
            phase = "PARAMAGNETICO"
            desc = "Desordem Paramagnética - Ruído"
        else:
            phase = "CRITICO"
            desc = "Ponto Crítico - Transição"

        return {
            'phase': phase,
            'description': desc,
            'direction': np.sign(magnetization)
        }


class ProjetorHolograficoMaldacena:
    """
    Implementação completa do Projetor Holográfico de Maldacena (PHM)

    Módulos:
    1. Tensorização do Estado do Mercado (Ψ) - Spin Chain + MPS
    2. Renormalização do Espaço (MERA) - Bulk AdS
    3. Fórmula de Ryu-Takayanagi - Entropia Holográfica
    4. Complexidade Holográfica - Volume do Bulk
    5. Output e Execução - Sinais de Trading
    """

    def __init__(self, window_size: int = 128, bond_dim: int = 8,
                 n_layers: int = 4, svd_cutoff: float = 1e-5):
        self.window_size = window_size
        self.bond_dim = bond_dim

        self.encoder = SpinChainEncoder()
        self.mera = SimplifiedMERA(bond_dim, n_layers, svd_cutoff)
        self.rt = RyuTakayanagi()
        self.complexity = HolographicComplexity()
        self.ising = IsingPhase()

        self._entropy_history = []
        self._complexity_history = []

    def analyze(self, prices: np.ndarray) -> dict:
        """Execução completa do PHM."""
        n = len(prices)
        if n < 10:
            return self._empty_result()

        window = prices[-self.window_size:] if n > self.window_size else prices

        # 1. Spin Chain + MPS
        spins = self.encoder.prices_to_spin_chain(window)
        mps = self.encoder.create_mps(spins, self.bond_dim)

        # 2. MERA
        mera_result = self.mera.build_mera(mps)

        # 3. Entropia de Ryu-Takayanagi
        entropy_profile = self.rt.compute_entropy_profile(mps)
        current_entropy = np.mean(entropy_profile) if len(entropy_profile) > 0 else 0.0

        spectra_entropies = [self.rt.entropy_from_spectrum(s)
                            for s in mera_result['entanglement_spectra'] if len(s) > 0]
        bulk_entropy = np.mean(spectra_entropies) if spectra_entropies else 0.0

        self._entropy_history.append(current_entropy)
        if len(self._entropy_history) > 50:
            self._entropy_history = self._entropy_history[-50:]

        spike = self.rt.detect_spike(np.array(self._entropy_history))

        # 4. Complexidade
        state_complexity = self.complexity.compute_complexity(mera_result)
        spectral_complexity = self.complexity.from_spectra(mera_result['entanglement_spectra'])
        total_complexity = state_complexity + spectral_complexity

        self._complexity_history.append(total_complexity)
        if len(self._complexity_history) > 50:
            self._complexity_history = self._complexity_history[-50:]

        stress = self.complexity.detect_stress(
            np.array(self._complexity_history),
            np.array(self._entropy_history)
        )

        # 5. Fase de Ising
        mag = self.ising.magnetization(spins)
        corr = self.ising.correlation(spins)
        phase = self.ising.determine_phase(mag, corr)

        # 6. Gerar sinal
        signal_result = self._generate_signal(spike, stress, phase)

        return {
            'signal': signal_result['signal'],
            'signal_name': signal_result['signal_name'],
            'confidence': signal_result['confidence'],
            'reasons': signal_result['reasons'],

            'entropy': current_entropy,
            'bulk_entropy': bulk_entropy,
            'entropy_profile': entropy_profile,
            'horizon_forming': spike['spike_detected'],
            'spike_magnitude': spike['magnitude'],

            'complexity': total_complexity,
            'computational_stress': stress,

            'phase': phase,
            'magnetization': mag,
            'correlation': corr,
            'phase_type': phase['phase'],
            'phase_direction': phase['direction'],

            'mera_layers': mera_result['n_layers'],
            'entanglement_spectra': mera_result['entanglement_spectra'],

            'n_observations': n,
            'current_price': prices[-1]
        }

    def _generate_signal(self, spike: dict, stress: dict, phase: dict) -> dict:
        """Gera sinal de trading."""
        signal = 0
        signal_name = "NEUTRO"
        confidence = 0.0
        reasons = []

        conditions = 0

        if spike['spike_detected']:
            conditions += 1
            reasons.append("Horizonte detectado (spike entropia)")

        if stress['stress_detected']:
            conditions += 0.5
            reasons.append("Stress computacional")

        p = phase['phase']
        direction = phase['direction']

        if p == "FERROMAGNETICO":
            if direction > 0:
                reasons.append("Ordem Ferromagnética BULLISH")
                preferred = 1
            else:
                reasons.append("Ordem Ferromagnética BEARISH")
                preferred = -1
        elif p == "PARAMAGNETICO":
            reasons.append("Desordem Paramagnética")
            preferred = -1
        else:
            reasons.append("Ponto Crítico")
            preferred = 0

        if conditions >= 1 and spike['spike_detected']:
            signal = preferred
            if signal == 1:
                signal_name = "LONG (Horizonte + Ferromagnético)"
            elif signal == -1:
                signal_name = "SHORT (Horizonte + Paramagnético)"
            else:
                signal_name = "ALERTA (Horizonte + Crítico)"
            confidence = min(conditions / 1.5, 1.0)
        elif conditions >= 0.5:
            signal_name = "ALERTA (Pré-horizonte)"
            confidence = 0.3

        return {
            'signal': signal,
            'signal_name': signal_name,
            'confidence': confidence,
            'reasons': reasons
        }

    def _empty_result(self) -> dict:
        """Retorna resultado vazio quando não há dados suficientes"""
        return {
            'signal': 0,
            'signal_name': 'HOLD',
            'confidence': 0.0,
            'reasons': ['Dados insuficientes'],
            'entropy': 0.0,
            'bulk_entropy': 0.0,
            'entropy_profile': np.array([]),
            'horizon_forming': False,
            'spike_magnitude': 0.0,
            'complexity': 0.0,
            'computational_stress': {'stress_detected': False, 'level': 0.0},
            'phase': {'phase': 'UNKNOWN', 'description': 'N/A', 'direction': 0},
            'magnetization': 0.0,
            'correlation': 0.0,
            'phase_type': 'UNKNOWN',
            'phase_direction': 0,
            'mera_layers': 0,
            'entanglement_spectra': [],
            'n_observations': 0,
            'current_price': 0.0
        }

    def get_signal(self, prices: np.ndarray) -> int:
        return self.analyze(prices)['signal']


def plot_phm_analysis(prices: np.ndarray, save_path: str = None):
    """Visualização do PHM."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("matplotlib não disponível")
        return None

    try:
        import networkx as nx
        HAS_NETWORKX = True
    except ImportError:
        HAS_NETWORKX = False

    phm = ProjetorHolograficoMaldacena(window_size=64, bond_dim=4, n_layers=3)
    result = phm.analyze(prices)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2)

    time = np.arange(len(prices))

    # Plot 1: Rede MERA
    ax1 = fig.add_subplot(gs[0, 0])

    if HAS_NETWORKX:
        G = nx.Graph()
        n_base = min(16, len(prices) - 1)
        positions = {}
        colors = []

        for i in range(n_base):
            G.add_node(f"L0_{i}")
            positions[f"L0_{i}"] = (i, 0)
            colors.append('lightblue')

        current_size = n_base
        for layer in range(1, result['mera_layers']):
            next_size = max(1, current_size // 2)
            for i in range(next_size):
                G.add_node(f"L{layer}_{i}")
                positions[f"L{layer}_{i}"] = (i * (n_base / next_size), layer * 2)

                spec_idx = min(i + (layer-1) * next_size,
                              len(result['entanglement_spectra']) - 1)
                if spec_idx >= 0 and len(result['entanglement_spectra']) > spec_idx:
                    colors.append('salmon')
                else:
                    colors.append('lightgray')

                p1 = min(i * 2, current_size - 1)
                p2 = min(i * 2 + 1, current_size - 1)
                G.add_edge(f"L{layer-1}_{p1}", f"L{layer}_{i}")
                if p1 != p2:
                    G.add_edge(f"L{layer-1}_{p2}", f"L{layer}_{i}")

            current_size = next_size

        nx.draw(G, positions, ax=ax1, node_color=colors[:len(G.nodes())],
               node_size=80, edge_color='gray', alpha=0.7)
    else:
        # Fallback visualization without networkx
        ax1.text(0.5, 0.5, f'MERA Network\n{result["mera_layers"]} layers',
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightblue'))

    ax1.set_title('Rede Tensorial MERA (Bulk AdS)', fontsize=11)

    # Plot 2: Preço + Fase
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time, prices, 'b-', linewidth=1.5)

    phase = result['phase_type']
    if phase == "FERROMAGNETICO":
        ax2.axvspan(0, len(prices), alpha=0.1, color='green')
    elif phase == "PARAMAGNETICO":
        ax2.axvspan(0, len(prices), alpha=0.1, color='red')
    else:
        ax2.axvspan(0, len(prices), alpha=0.1, color='yellow')

    if result['signal'] == 1:
        ax2.scatter([time[-1]], [prices[-1]], c='green', s=200, marker='^')
    elif result['signal'] == -1:
        ax2.scatter([time[-1]], [prices[-1]], c='red', s=200, marker='v')

    ax2.set_title(f'Preço | Fase: {phase} | M={result["magnetization"]:.3f}', fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Entropia
    ax3 = fig.add_subplot(gs[1, 0])
    if len(result['entropy_profile']) > 0:
        ax3.plot(result['entropy_profile'], 'purple', linewidth=1.5)
        ax3.fill_between(range(len(result['entropy_profile'])), 0,
                        result['entropy_profile'], alpha=0.3, color='purple')

    if result['horizon_forming']:
        ax3.text(0.5, 0.9, 'HORIZONTE FORMANDO', transform=ax3.transAxes,
                ha='center', color='red', fontsize=12, weight='bold')

    ax3.set_title('Entropia de Ryu-Takayanagi S_A', fontsize=11)
    ax3.set_xlabel('Posição de Corte')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Complexidade vs Entropia
    ax4 = fig.add_subplot(gs[1, 1])
    if len(phm._complexity_history) > 1:
        ax4.plot(phm._complexity_history, 'orange', label='Complexidade')
        ax4_t = ax4.twinx()
        ax4_t.plot(phm._entropy_history, 'purple', alpha=0.7, label='Entropia')
        ax4_t.set_ylabel('Entropia', color='purple')
    ax4.set_title('Complexidade vs Entropia', fontsize=11)
    ax4.set_ylabel('Complexidade', color='orange')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Espectros de entrelaçamento
    ax5 = fig.add_subplot(gs[2, 0])
    for i, spec in enumerate(result['entanglement_spectra'][:5]):
        if len(spec) > 0:
            ax5.semilogy(spec, label=f'Camada {i+1}', alpha=0.7)
    ax5.axhline(y=1e-5, color='red', linestyle='--', label='Cutoff SVD')
    ax5.set_title('Espectros de Entrelaçamento', fontsize=11)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Diagrama de fase
    ax6 = fig.add_subplot(gs[2, 1])
    J = np.linspace(0, 2, 30)
    h = np.linspace(0, 2, 30)
    JJ, hh = np.meshgrid(J, h)
    phase_map = np.where(JJ > hh, 1, -1)
    ax6.contourf(J, h, phase_map, levels=[-1, 0, 1],
                colors=['lightcoral', 'white', 'lightgreen'], alpha=0.7)
    ax6.contour(J, h, phase_map, levels=[0], colors='black', linewidths=2)
    ax6.scatter([1.0], [0.5], c='blue', s=150, marker='*')
    ax6.set_xlabel('J (Acoplamento)')
    ax6.set_ylabel('h (Campo)')
    ax6.set_title('Diagrama de Fase Ising', fontsize=11)

    # Resumo
    summary = (f"PHM | Sinal: {result['signal_name']} | "
              f"Fase: {result['phase_type']} | "
              f"S={result['entropy']:.3f} | C={result['complexity']:.1f}")

    color = 'green' if result['signal'] == 1 else 'red' if result['signal'] == -1 else 'lightblue'
    fig.text(0.5, 0.01, summary, fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    print("=" * 70)
    print("PROJETOR HOLOGRAFICO DE MALDACENA (PHM)")
    print("Física de Altas Energias / Teoria da Informação Quântica")
    print("=" * 70)

    np.random.seed(42)

    # Dados simulados
    trend = 1.1 + 0.0003 * np.cumsum(np.ones(80) + 0.2 * np.random.randn(80))
    critical = trend[-1] + 0.0001 * np.cumsum(np.random.randn(60))
    disorder = critical[-1] + 0.0004 * np.cumsum(np.random.randn(60))

    prices = np.concatenate([trend, critical, disorder])

    print(f"\nDados: {len(prices)} pontos")
    print(f"Preço: {prices[0]:.5f} -> {prices[-1]:.5f}")

    phm = ProjetorHolograficoMaldacena(window_size=64, bond_dim=4, n_layers=3)

    print("\n" + "-" * 40)
    print("Executando análise PHM...")
    print("-" * 40)

    result = phm.analyze(prices)

    print(f"\n📊 SINAL: {result['signal_name']}")
    print(f"   Confiança: {result['confidence']:.0%}")

    print(f"\n🌌 ENTROPIA RYU-TAKAYANAGI:")
    print(f"   S_A: {result['entropy']:.4f}")
    print(f"   Bulk: {result['bulk_entropy']:.4f}")
    print(f"   Horizonte: {'SIM' if result['horizon_forming'] else 'NAO'}")
    print(f"   Spike Magnitude: {result['spike_magnitude']:.3f}")

    print(f"\n🔮 COMPLEXIDADE HOLOGRAFICA:")
    print(f"   C: {result['complexity']:.1f}")
    print(f"   Stress: {'SIM' if result['computational_stress']['stress_detected'] else 'NAO'}")
    print(f"   Level: {result['computational_stress']['level']:.3f}")

    print(f"\n⚡ FASE ISING:")
    print(f"   Tipo: {result['phase_type']}")
    print(f"   Magnetização: {result['magnetization']:.4f}")
    print(f"   Correlação: {result['correlation']:.4f}")
    print(f"   Direção: {result['phase_direction']}")

    print(f"\n📝 RAZOES:")
    for reason in result['reasons']:
        print(f"   • {reason}")

    print("\n" + "=" * 70)
    if result['signal'] == 1:
        print("LONG - ORDEM FERROMAGNETICA + HORIZONTE!")
    elif result['signal'] == -1:
        print("SHORT - DESORDEM PARAMAGNETICA!")
    elif result['horizon_forming']:
        print("HORIZONTE DE EVENTOS FORMANDO!")
    else:
        print("NEUTRO - Espaço-tempo estável")
    print("=" * 70)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        print("\nGerando visualização...")
        plot_phm_analysis(prices, '/tmp/phm_analysis.png')
        print("Visualização salva: /tmp/phm_analysis.png")
        plt.close()
    except Exception as e:
        print(f"Erro na visualização: {e}")

    print("\n✅ Teste do PHM concluído com sucesso!")
