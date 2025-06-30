import json
import time
import numpy as np
from typing import Dict, List, Union, Tuple, Complex, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum, auto

# --- Estados y Enumeraciones ---
class EstadoQubit(Enum):
    GROUND = auto()
    EXCITED = auto()
    SUPERPOSITION = auto()

class OperacionCuantica(Enum):
    RESET = auto()
    HADAMARD = auto()
    ROTACION_X = auto()
    ROTACION_Y = auto()
    ROTACION_Z = auto()
    MEASUREMENT = auto()
    COLLAPSE = auto()

class TipoDecaimiento(Enum):
    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"
    FISSION = "fission"
    BETA_GAMMA = "beta_gamma"
    ALPHA_SF = "alpha_sf"

# --- Clases de Datos Estructurados ---
@dataclass
class EstadoComplejo:
    """Representación normalizada de un estado cuántico de un qubit."""
    alpha: complex = 1.0 + 0.0j
    beta: complex = 0.0 + 0.0j
    
    def __post_init__(self):
        self.normalize()
    
    def normalize(self):
        norm = np.sqrt(abs(self.alpha)**2 + abs(self.beta)**2)
        if norm > 1e-9:
            self.alpha /= norm
            self.beta /= norm
    
    @property
    def vector(self) -> np.ndarray:
        return np.array([self.alpha, self.beta], dtype=complex)
    
    def __str__(self):
        return f"({self.alpha:.3f})|0⟩ + ({self.beta:.3f})|1⟩"

@dataclass
class EstadoFoton:
    """Estado de un fotón óptico usado para la comunicación."""
    polarizacion: float  # Ángulo en radianes
    fase: float         # Fase en radianes
    valido: bool = True # False si el fotón se perdió o corrompió

@dataclass
class FirmaRadiactiva:
    """Define la firma de radiación que 'colorea' la transmisión."""
    isotopo: str
    energia_pico_ev: float
    tipo_decaimiento: TipoDecaimiento
    vida_media_s: float
    spin_nuclear: float = 0.0

@dataclass
class PulsoMicroondas:
    """Parámetros para un pulso de control físico."""
    operacion: OperacionCuantica
    angulo: float = 0.0
    duracion_ns: float = 20.0
    amplitud_V: float = 0.5
    fase_rad: float = 0.0

@dataclass
class QuantumRadiationState:
    """Estado cuántico-radiactivo unificado para comunicación."""
    isotope: str
    energy_level: float
    decay_rate: float
    spin_state: EstadoComplejo
    entanglement_phase: float
    coherence_time: float
    firma_radiactiva: FirmaRadiactiva

@dataclass(frozen=True)
class PaqueteBiMoType:
    """El paquete de datos completo que se transmite."""
    id_mensaje: str
    timestamp: float
    carga_util: Dict[str, Any]
    firma_radiactiva: FirmaRadiactiva
    estados_cuanticos: List[QuantumRadiationState] = field(default_factory=list)

@dataclass
class MetricasSistema:
    """Métricas de estado del sistema en un ciclo de simulación."""
    ciclo: int
    temperatura_mk: float
    decoherencia_t2_us: float
    qber_estimado: float  # Quantum Bit Error Rate
    perdida_canal_db: float
    potencia_ruido_dbm: float
    fidelidad_promedio: float = 0.0

class SistemaQuantumBiMoType:
    """
    Sistema unificado para comunicación cuántica-radiactiva BiMOtype
    Integra protocolo de comunicación, estados cuánticos y métricas del sistema
    """
    
    def __init__(self):
        # Elementos radiactivos con información completa
        self.radioactive_elements = {
            'U235': {'half_life': 703800000, 'energy': 202.5, 'type': TipoDecaimiento.FISSION, 'spin': 7/2},
            'U238': {'half_life': 4468000000, 'energy': 4.27, 'type': TipoDecaimiento.ALPHA, 'spin': 0},
            'Pu239': {'half_life': 24110, 'energy': 200.0, 'type': TipoDecaimiento.FISSION, 'spin': 1/2},
            'Pu238': {'half_life': 87.7, 'energy': 5.59, 'type': TipoDecaimiento.ALPHA, 'spin': 0},
            'Th232': {'half_life': 14050000000, 'energy': 4.08, 'type': TipoDecaimiento.ALPHA, 'spin': 0},
            'Sr90': {'half_life': 28.8, 'energy': 0.546, 'type': TipoDecaimiento.BETA, 'spin': 0},
            'Co60': {'half_life': 5.27, 'energy': 2.82, 'type': TipoDecaimiento.BETA_GAMMA, 'spin': 5},
            'Cm244': {'half_life': 18.1, 'energy': 5.81, 'type': TipoDecaimiento.ALPHA, 'spin': 0},
            'Po210': {'half_life': 0.38, 'energy': 5.41, 'type': TipoDecaimiento.ALPHA, 'spin': 0},
            'Am241': {'half_life': 432.6, 'energy': 5.49, 'type': TipoDecaimiento.ALPHA, 'spin': 5/2},
            'Cf252': {'half_life': 2.65, 'energy': 6.12, 'type': TipoDecaimiento.ALPHA_SF, 'spin': 0},
            'Tc99m': {'half_life': 0.25, 'energy': 0.14, 'type': TipoDecaimiento.GAMMA, 'spin': 9/2}
        }
        
        # Mapeo Morse-Cuántico mejorado
        self.morse_quantum_map = {
            'A': {'morse': '.-', 'quantum': '|0⟩|1⟩', 'isotope': 'Sr90', 'phase': 0},
            'B': {'morse': '-...', 'quantum': '|1⟩|0⟩|0⟩|0⟩', 'isotope': 'Co60', 'phase': np.pi/4},
            'C': {'morse': '-.-.', 'quantum': '|1⟩|0⟩|1⟩|0⟩', 'isotope': 'Pu238', 'phase': np.pi/2},
            'D': {'morse': '-..', 'quantum': '|1⟩|0⟩|0⟩', 'isotope': 'U235', 'phase': np.pi/3},
            'E': {'morse': '.', 'quantum': '|0⟩', 'isotope': 'Tc99m', 'phase': 0},
            'F': {'morse': '..-.', 'quantum': '|0⟩|0⟩|1⟩|0⟩', 'isotope': 'Am241', 'phase': np.pi/6},
            'G': {'morse': '--.', 'quantum': '|1⟩|1⟩|0⟩', 'isotope': 'Cm244', 'phase': np.pi/5},
            'H': {'morse': '....', 'quantum': '|0⟩|0⟩|0⟩|0⟩', 'isotope': 'Po210', 'phase': 0},
            'I': {'morse': '..', 'quantum': '|0⟩|0⟩', 'isotope': 'Sr90', 'phase': np.pi/8},
            'J': {'morse': '.---', 'quantum': '|0⟩|1⟩|1⟩|1⟩', 'isotope': 'U238', 'phase': np.pi/7},
            'K': {'morse': '-.-', 'quantum': '|1⟩|0⟩|1⟩', 'isotope': 'Pu239', 'phase': np.pi/4},
            'L': {'morse': '.-..', 'quantum': '|0⟩|1⟩|0⟩|0⟩', 'isotope': 'Th232', 'phase': np.pi/3},
            'M': {'morse': '--', 'quantum': '|1⟩|1⟩', 'isotope': 'Cf252', 'phase': np.pi/2},
            'N': {'morse': '-.', 'quantum': '|1⟩|0⟩', 'isotope': 'Co60', 'phase': np.pi/6},
            'O': {'morse': '---', 'quantum': '|1⟩|1⟩|1⟩', 'isotope': 'U235', 'phase': 2*np.pi/3},
            'P': {'morse': '.--.', 'quantum': '|0⟩|1⟩|1⟩|0⟩', 'isotope': 'Am241', 'phase': np.pi/5},
            'Q': {'morse': '--.-', 'quantum': '|1⟩|1⟩|0⟩|1⟩', 'isotope': 'Pu238', 'phase': 3*np.pi/4},
            'R': {'morse': '.-.', 'quantum': '|0⟩|1⟩|0⟩', 'isotope': 'Sr90', 'phase': np.pi/4},
            'S': {'morse': '...', 'quantum': '|0⟩|0⟩|0⟩', 'isotope': 'Tc99m', 'phase': 0},
            'T': {'morse': '-', 'quantum': '|1⟩', 'isotope': 'Co60', 'phase': np.pi},
            'U': {'morse': '..-', 'quantum': '|0⟩|0⟩|1⟩', 'isotope': 'U238', 'phase': np.pi/3},
            'V': {'morse': '...-', 'quantum': '|0⟩|0⟩|0⟩|1⟩', 'isotope': 'Cm244', 'phase': np.pi/7},
            'W': {'morse': '.--', 'quantum': '|0⟩|1⟩|1⟩', 'isotope': 'Pu239', 'phase': 2*np.pi/3},
            'X': {'morse': '-..-', 'quantum': '|1⟩|0⟩|0⟩|1⟩', 'isotope': 'Po210', 'phase': 3*np.pi/5},
            'Y': {'morse': '-.--', 'quantum': '|1⟩|0⟩|1⟩|1⟩', 'isotope': 'Cf252', 'phase': 4*np.pi/5},
            'Z': {'morse': '--..', 'quantum': '|1⟩|1⟩|0⟩|0⟩', 'isotope': 'Th232', 'phase': np.pi/2},
            '0': {'morse': '-----', 'quantum': '|00000⟩', 'isotope': 'U235', 'phase': 0},
            '1': {'morse': '.----', 'quantum': '|00001⟩', 'isotope': 'Pu239', 'phase': np.pi/5},
            '2': {'morse': '..---', 'quantum': '|00011⟩', 'isotope': 'Th232', 'phase': 2*np.pi/5},
            '3': {'morse': '...--', 'quantum': '|00111⟩', 'isotope': 'U238', 'phase': 3*np.pi/5},
            '4': {'morse': '....-', 'quantum': '|01111⟩', 'isotope': 'Am241', 'phase': 4*np.pi/5},
            '5': {'morse': '.....', 'quantum': '|11111⟩', 'isotope': 'Sr90', 'phase': np.pi},
            '6': {'morse': '-....', 'quantum': '|11110⟩', 'isotope': 'Co60', 'phase': 6*np.pi/5},
            '7': {'morse': '--...', 'quantum': '|11100⟩', 'isotope': 'Cm244', 'phase': 7*np.pi/5},
            '8': {'morse': '---..', 'quantum': '|11000⟩', 'isotope': 'Po210', 'phase': 8*np.pi/5},
            '9': {'morse': '----.', 'quantum': '|10000⟩', 'isotope': 'Cf252', 'phase': 9*np.pi/5},
            ' ': {'morse': '/', 'quantum': '⊗', 'isotope': 'vacuum', 'phase': 0},
        }
        
        # Inicializar métricas del sistema
        self.metricas = MetricasSistema(
            ciclo=0,
            temperatura_mk=15.0,
            decoherencia_t2_us=100.0,
            qber_estimado=0.01,
            perdida_canal_db=0.5,
            potencia_ruido_dbm=-90.0
        )
    
    def crear_estado_cuantico(self, char_data: Dict, isotope: str) -> QuantumRadiationState:
        """Crea un estado cuántico-radiactivo unificado."""
        phase = char_data['phase']
        estado_complejo = EstadoComplejo(
            alpha=complex(np.cos(phase), 0),
            beta=complex(np.sin(phase), 0)
        )
        
        isotope_data = self.radioactive_elements.get(isotope, {})
        firma_radiactiva = FirmaRadiactiva(
            isotopo=isotope,
            energia_pico_ev=isotope_data.get('energy', 1.0),
            tipo_decaimiento=isotope_data.get('type', TipoDecaimiento.ALPHA),
            vida_media_s=isotope_data.get('half_life', 1.0) * 3.154e7,  # años a segundos
            spin_nuclear=isotope_data.get('spin', 0.0)
        )
        
        return QuantumRadiationState(
            isotope=isotope,
            energy_level=isotope_data.get('energy', 1.0),
            decay_rate=1.0/isotope_data.get('half_life', 1.0),
            spin_state=estado_complejo,
            entanglement_phase=phase,
            coherence_time=self.calculate_coherence_time(isotope),
            firma_radiactiva=firma_radiactiva
        )
    
    def encode_quantum_message(self, message: str) -> PaqueteBiMoType:
        """Codifica un mensaje en un paquete BiMOtype unificado."""
        timestamp = time.time()
        estados_cuanticos = []
        quantum_states_data = []
        
        for i, char in enumerate(message.upper()):
            if char in self.morse_quantum_map:
                char_data = self.morse_quantum_map[char]
                isotope = char_data['isotope']
                
                if isotope == 'vacuum':
                    continue
                
                quantum_state = self.crear_estado_cuantico(char_data, isotope)
                estados_cuanticos.append(quantum_state)
                
                quantum_states_data.append({
                    'position': i,
                    'character': char,
                    'morse': char_data['morse'],
                    'quantum_state': char_data['quantum'],
                    'isotope': isotope,
                    'phase': char_data['phase'],
                    'estado_complejo': str(quantum_state.spin_state),
                    'energia': quantum_state.energy_level
                })
        
        # Crear firma radiactiva promedio
        if estados_cuanticos:
            energia_promedio = np.mean([e.energy_level for e in estados_cuanticos])
            isotopo_dominante = max(set(e.isotope for e in estados_cuanticos), 
                                  key=lambda x: sum(1 for e in estados_cuanticos if e.isotope == x))
            firma_dominante = estados_cuanticos[0].firma_radiactiva
        else:
            energia_promedio = 0.0
            isotopo_dominante = 'vacuum'
            firma_dominante = FirmaRadiactiva('vacuum', 0.0, TipoDecaimiento.ALPHA, 0.0)
        
        carga_util = {
            "protocolo": "BiMOtype-Quantum-Unified-v4.0",
            "mensaje_original": message,
            "estados_cuanticos": quantum_states_data,
            "estadisticas": {
                "total_qubits": len(estados_cuanticos),
                "energia_total": sum(e.energy_level for e in estados_cuanticos),
                "tiempo_coherencia_promedio": np.mean([e.coherence_time for e in estados_cuanticos]) if estados_cuanticos else 0
            }
        }
        
        return PaqueteBiMoType(
            id_mensaje=f"BiMO-{int(timestamp)}-{hash(message) % 10000:04d}",
            timestamp=timestamp,
            carga_util=carga_util,
            firma_radiactiva=firma_dominante,
            estados_cuanticos=estados_cuanticos
        )
    
    def aplicar_operacion_cuantica(self, estado: EstadoComplejo, operacion: OperacionCuantica, 
                                  parametros: Dict = None) -> EstadoComplejo:
        """Aplica operaciones cuánticas a un estado."""
        if parametros is None:
            parametros = {}
        
        if operacion == OperacionCuantica.HADAMARD:
            # Puerta Hadamard: |0⟩ → (|0⟩ + |1⟩)/√2, |1⟩ → (|0⟩ - |1⟩)/√2
            nueva_alpha = (estado.alpha + estado.beta) / np.sqrt(2)
            nueva_beta = (estado.alpha - estado.beta) / np.sqrt(2)
            return EstadoComplejo(nueva_alpha, nueva_beta)
        
        elif operacion == OperacionCuantica.ROTACION_X:
            angulo = parametros.get('angulo', np.pi/2)
            cos_half = np.cos(angulo/2)
            sin_half = 1j * np.sin(angulo/2)
            nueva_alpha = cos_half * estado.alpha + sin_half * estado.beta
            nueva_beta = sin_half * estado.alpha + cos_half * estado.beta
            return EstadoComplejo(nueva_alpha, nueva_beta)
        
        elif operacion == OperacionCuantica.ROTACION_Z:
            angulo = parametros.get('angulo', np.pi/2)
            fase = np.exp(1j * angulo/2)
            nueva_alpha = estado.alpha * np.conj(fase)
            nueva_beta = estado.beta * fase
            return EstadoComplejo(nueva_alpha, nueva_beta)
        
        elif operacion == OperacionCuantica.RESET:
            return EstadoComplejo(1.0, 0.0)
        
        else:
            return estado
    
    def simular_decoherencia(self, estado: EstadoComplejo, tiempo_ns: float) -> EstadoComplejo:
        """Simula efectos de decoherencia en un estado cuántico."""
        T2 = self.metricas.decoherencia_t2_us * 1000  # convertir a ns
        factor_decoherencia = np.exp(-tiempo_ns / T2)
        
        # Aplicar decoherencia principalmente a la fase
        nueva_alpha = estado.alpha
        nueva_beta = estado.beta * factor_decoherencia
        
        # Añadir ruido térmico
        ruido_termico = np.sqrt(self.metricas.temperatura_mk / 1000.0) * 0.01
        nueva_alpha += np.random.normal(0, ruido_termico) * (1 + 1j)
        nueva_beta += np.random.normal(0, ruido_termico) * (1 + 1j)
        
        return EstadoComplejo(nueva_alpha, nueva_beta)
    
    def decode_quantum_transmission(self, paquete: PaqueteBiMoType, 
                                  mediciones_ruidosas: List[Dict]) -> Dict:
        """Decodifica una transmisión cuántica con métricas avanzadas."""
        try:
            decoded_chars = []
            fidelidades = []
            errores = []
            
            for i, medicion in enumerate(mediciones_ruidosas):
                # Reconstruir estado cuántico a partir de la medición
                estado_reconstruido = self.reconstruir_estado_cuantico(medicion)
                
                # Encontrar la mejor coincidencia
                mejor_coincidencia, mejor_fidelidad = self.encontrar_mejor_coincidencia(estado_reconstruido)
                
                if mejor_fidelidad > 0.7:
                    decoded_chars.append(mejor_coincidencia)
                    fidelidades.append(mejor_fidelidad)
                else:
                    decoded_chars.append('?')
                    fidelidades.append(0.0)
                    errores.append(f"Posición {i}: Baja fidelidad ({mejor_fidelidad:.2f})")
            
            fidelidad_promedio = np.mean(fidelidades) if fidelidades else 0.0
            self.metricas.fidelidad_promedio = fidelidad_promedio
            
            # Actualizar QBER
            errores_cuanticos = sum(1 for f in fidelidades if f < 0.9)
            self.metricas.qber_estimado = errores_cuanticos / len(fidelidades) if fidelidades else 1.0
            
            return {
                "estado": "EXITO" if fidelidad_promedio > 0.8 else "DEGRADADO",
                "mensaje_decodificado": ''.join(decoded_chars),
                "metricas_cuanticas": {
                    "fidelidad_promedio": fidelidad_promedio,
                    "qber": self.metricas.qber_estimado,
                    "decoherencia_detectada": any(f < 0.5 for f in fidelidades)
                },
                "errores": errores,
                "calidad_reconstruccion": self.evaluar_calidad(fidelidad_promedio),
                "metricas_sistema": self.metricas
            }
        except Exception as e:
            return {"estado": "ERROR", "mensaje_error": str(e)}
    
    def calculate_coherence_time(self, isotope: str) -> float:
        """Calcula el tiempo de coherencia basado en el isótopo."""
        if isotope in self.radioactive_elements:
            half_life = self.radioactive_elements[isotope]['half_life']
            return min(1e-3, 1.0 / (np.log(2) / (half_life * 3.154e7)))
        return 1e-6
    
    def reconstruir_estado_cuantico(self, medicion: Dict) -> EstadoComplejo:
        """Reconstruye un estado cuántico a partir de una medición."""
        energia = medicion.get('energia_medida', 0)
        fase_estimada = (energia % 1.0) * 2 * np.pi
        
        alpha = complex(np.cos(fase_estimada/2), 0)
        beta = complex(np.sin(fase_estimada/2), 0)
        
        return EstadoComplejo(alpha, beta)
    
    def encontrar_mejor_coincidencia(self, estado: EstadoComplejo) -> Tuple[str, float]:
        """Encuentra la mejor coincidencia para un estado cuántico."""
        mejor_char = '?'
        mejor_fidelidad = 0.0
        
        for char, char_data in self.morse_quantum_map.items():
            if char == ' ':
                continue
            
            # Crear estado ideal
            fase_ideal = char_data['phase']
            estado_ideal = EstadoComplejo(
                complex(np.cos(fase_ideal/2), 0),
                complex(np.sin(fase_ideal/2), 0)
            )
            
            # Calcular fidelidad cuántica
            fidelidad = self.calcular_fidelidad_cuantica(estado, estado_ideal)
            
            if fidelidad > mejor_fidelidad:
                mejor_fidelidad = fidelidad
                mejor_char = char
        
        return mejor_char, mejor_fidelidad
    
    def calcular_fidelidad_cuantica(self, estado1: EstadoComplejo, estado2: EstadoComplejo) -> float:
        """Calcula la fidelidad cuántica entre dos estados."""
        # Fidelidad = |⟨ψ₁|ψ₂⟩|²
        producto_escalar = np.conj(estado1.alpha) * estado2.alpha + np.conj(estado1.beta) * estado2.beta
        return abs(producto_escalar) ** 2
    
    def evaluar_calidad(self, fidelidad: float) -> str:
        """Evalúa la calidad de la reconstrucción."""
        if fidelidad > 0.95:
            return "EXCELENTE"
        elif fidelidad > 0.85:
            return "BUENA"
        elif fidelidad > 0.70:
            return "ACEPTABLE"
        else:
            return "POBRE"
    
    def actualizar_metricas_sistema(self):
        """Actualiza las métricas del sistema para el siguiente ciclo."""
        self.metricas.ciclo += 1
        # Simular fluctuaciones térmicas
        self.metricas.temperatura_mk += np.random.normal(0, 0.5)
        self.metricas.temperatura_mk = max(10.0, self.metricas.temperatura_mk)
        
        # Actualizar tiempo de decoherencia basado en temperatura
        self.metricas.decoherencia_t2_us = 200.0 / (self.metricas.temperatura_mk / 15.0)
        
        # Simular variaciones en el ruido del canal
        self.metricas.potencia_ruido_dbm += np.random.normal(0, 1.0)

# --- Demostración del Sistema Unificado ---
if __name__ == "__main__":
    print("🚀 SISTEMA UNIFICADO BiMOtype QUANTUM OS 🚀")
    print("=" * 60)
    
    # Inicializar sistema
    sistema = SistemaQuantumBiMoType()
    mensaje = "HELLO QUANTUM WORLD"
    
    print(f"Mensaje a transmitir: '{mensaje}'")
    print("-" * 60)
    
    # Codificar mensaje
    print("\n[CODIFICACIÓN]")
    paquete = sistema.encode_quantum_message(mensaje)
    print(f"✅ Paquete creado: {paquete.id_mensaje}")
    print(f"📊 Estados cuánticos generados: {len(paquete.estados_cuanticos)}")
    print(f"🔬 Firma radiactiva dominante: {paquete.firma_radiactiva.isotopo}")
    
    # Simular transmisión con ruido
    print("\n[TRANSMISIÓN Y MEDICIÓN]")
    mediciones_simuladas = []
    for estado in paquete.estados_cuanticos:
        # Simular ruido en la medición
        energia_ruidosa = estado.energy_level + np.random.normal(0, 0.1)
        mediciones_simuladas.append({'energia_medida': energia_ruidosa})
    
    # Simular efectos de decoherencia
    estados_decoherentes = []
    for estado in paquete.estados_cuanticos:
        estado_decoherente = sistema.simular_decoherencia(estado.spin_state, 50.0)  # 50 ns
        estados_decoherentes.append(estado_decoherente)
    
    print(f"✅ Simulación de {len(mediciones_simuladas)} mediciones completada")
    
    # Decodificar
    print("\n[DECODIFICACIÓN]")
    resultado = sistema.decode_quantum_transmission(paquete, mediciones_simuladas)
    
    # Mostrar resultados
    print("\n" + "=" * 60)
    print("📊 RESULTADOS FINALES")
    print("=" * 60)
    print(f"Mensaje Original:    '{mensaje}'")
    print(f"Mensaje Decodificado: '{resultado.get('mensaje_decodificado', 'ERROR')}'")
    print(f"Estado de Transmisión: {resultado.get('estado', 'DESCONOCIDO')}")
    
    metricas = resultado.get('metricas_cuanticas', {})
    print(f"\n📈 Métricas Cuánticas:")
    print(f"  Fidelidad Promedio: {metricas.get('fidelidad_promedio', 0):.1%}")
    print(f"  QBER: ('qber(tasa de error):  {metricas.get('qber', 0):.1%}")
    print(f"   • Estados decodificados: {metricas.get('estados_decodificados', 0)}")
    print(f"   • Decoherencia detectada: {'Sí' if metricas.get('decoherencia_detectada', False) else 'No'}")
    
    print(f"\n⚙️  MÉTRICAS DEL SISTEMA:")
    print(f"   • Ciclo de operación:    {sistema.get('ciclo', 0)}")
    print(f"   • Temperatura:           {sistema.get('temperatura_mk', 0)} mK")
    print(f"   • Tiempo coherencia T₂:  {sistema.get('tiempo_coherencia_t2', 0)} μs")
    
    if resultado.get('errores'):
        print(f"\n⚠️  ERRORES Y ADVERTENCIAS:")
        for error in resultado['errores']:
            print(f"   • {error}")
    
    print("\n" + "=" * 80)
    print("🎉 DEMOSTRACIÓN COMPLETADA")
    print("=" * 80)
