# BiMoType/datatypes.py

import numpy as np
from typing import Dict, List, Union, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum, auto

# --- Estados y Enumeraciones ---
class EstadoQubitConceptual(Enum): # Renombrado para evitar colisión
    """Estados conceptuales de un qubit."""
    GROUND = auto()
    EXCITED = auto()
    SUPERPOSITION = auto()

class OperacionCuantica(Enum):
    """Tipos de operaciones cuánticas aplicables a qubits."""
    RESET = auto()
    HADAMARD = auto()
    ROTACION_X = auto()
    ROTACION_Y = auto()
    ROTACION_Z = auto()
    MEASUREMENT = auto()
    COLLAPSE = auto()

class TipoDecaimiento(Enum):
    """Clasificación de los tipos de decaimiento radiactivo."""
    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"
    FISSION = "fission"
    BETA_GAMMA = "beta_gamma"
    ALPHA_SF = "alpha_sf" # Spontaneous Fission

# --- Clases de Datos Estructurados ---
@dataclass
class EstadoComplejo:
    """
    Representación normalizada del estado cuántico de un qubit (vector de estado).
    |ψ⟩ = α|0⟩ + β|1⟩
    """
    alpha: complex = 1.0 + 0.0j
    beta: complex = 0.0 + 0.0j

    def __post_init__(self):
        """Asegura que el estado esté normalizado después de la inicialización."""
        self.normalize()

    def normalize(self):
        """Normaliza el vector de estado para que su magnitud sea 1."""
        norm = np.sqrt(abs(self.alpha)**2 + abs(self.beta)**2)
        if norm > 1e-9: # Evitar división por cero o números muy pequeños
            self.alpha /= norm
            self.beta /= norm
        else: # Si la norma es muy pequeña, establecer a un estado predeterminado
            self.alpha = 1.0 + 0.0j
            self.beta = 0.0 + 0.0j


    @property
    def vector(self) -> np.ndarray:
        """Retorna el estado como un array de NumPy."""
        return np.array([self.alpha, self.beta], dtype=complex)

    def __str__(self):
        """Representación de cadena del estado complejo."""
        return f"({self.alpha:.3f})|0⟩ + ({self.beta:.3f})|1⟩"

@dataclass
class EstadoFoton:
    """Estado de un fotón óptico usado para la comunicación."""
    polarizacion: float  # Ángulo en radianes (e.g., 0 para horizontal, pi/2 para vertical)
    fase: float         # Fase en radianes
    valido: bool = True # False si el fotón se perdió o corrompió

@dataclass
class FirmaRadiactiva:
    """
    Define la firma de radiación que 'colorea' la transmisión,
    basada en propiedades de un isótopo y métricas del cuadrante-coremind.
    """
    isotopo: str
    energia_pico_ev: float # Energía dominante asociada
    tipo_decaimiento: TipoDecaimiento
    vida_media_s: float # Vida media en segundos
    spin_nuclear: float = 0.0
    # Añadidos para integrar el cuadrante-coremind
    mahalanobis_distance: Optional[float] = None
    lambda_double_non_locality: Optional[float] = None
    mg_polarity: Optional[float] = None
    mg_threshold: Optional[float] = None
    vacuum_polarity_n_r: Optional[float] = None # Polaridad del vacío n(r)

@dataclass
class PulsoMicroondas:
    """Parámetros para un pulso de control físico aplicado a un qubit."""
    operacion: OperacionCuantica
    angulo: float = 0.0      # Ángulo de rotación para R_X, R_Y, R_Z
    duracion_ns: float = 20.0 # Duración del pulso en nanosegundos
    amplitud_V: float = 0.5   # Amplitud del pulso en voltios
    fase_rad: float = 0.0     # Fase del pulso en radianes

@dataclass
class QuantumRadiationState:
    """
    Estado cuántico-radiactivo unificado para comunicación,
    representando el portador de información.
    """
    isotope: str
    energy_level: float         # Energía del estado de radiación
    decay_rate: float           # Tasa de decaimiento del isótopo (en 1/segundos)
    spin_state: EstadoComplejo  # Estado de spin del núcleo o de la partícula emitida
    entanglement_phase: float   # Fase de entrelazamiento con otros componentes o el vacío
    coherence_time: float       # Tiempo de coherencia esperado (en segundos)
    firma_radiactiva: FirmaRadiactiva # La firma específica de este estado

@dataclass(frozen=True)
class PaqueteBiMoType:
    """
    El paquete de datos completo que se transmite en el protocolo BiMoType.
    Contiene la carga útil clásica y la información cuántico-radiactiva.
    """
    id_mensaje: str
    timestamp: float # Marca de tiempo de creación del paquete
    carga_util: Dict[str, Any] # Los datos originales del mensaje o de la IA
    firma_radiactiva: FirmaRadiactiva # La firma general del paquete
    estados_cuanticos: List[QuantumRadiationState] = field(default_factory=list) # Lista de portadores de información

@dataclass
class MetricasSistema:
    """
    Métricas de estado del sistema en un ciclo de simulación,
    usadas para evaluar el rendimiento y la calidad de la transmisión.
    """
    ciclo: int = 0
    temperatura_mk: float = 15.0 # Temperatura en miliKelvin
    decoherencia_t2_us: float = 100.0 # Tiempo de decoherencia T2 en microsegundos
    qber_estimado: float = 0.01 # Quantum Bit Error Rate (Tasa de error de bit cuántico)
    perdida_canal_db: float = 0.5 # Pérdida de señal en el canal en decibelios
    potencia_ruido_dbm: float = -90.0 # Potencia de ruido en decibelios-milivatio
    fidelidad_promedio: float = 0.0 # Fidelidad promedio de la transmisión
    # Añadimos un campo para el tiempo de coherencia T1 si se modela más tarde
    decoherencia_t1_us: float = 200.0 # Tiempo de relajación T1 en microsegundos
