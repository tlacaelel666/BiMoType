# components/classic.py

import logging
import numpy as np
import cmath
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import threading
import time
from abc import ABC, abstractmethod

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importaciones del core (simuladas aquí para el ejemplo)
@dataclass
class EstadoComplejo:
    """Representa un estado cuántico complejo."""
    alpha: complex = 1.0 + 0j  # Amplitud para |0⟩
    beta: complex = 0.0 + 0j   # Amplitud para |1⟩
    
    def __post_init__(self):
        """Normaliza el estado después de la inicialización."""
        self.normalizar()
    
    def normalizar(self):
        """Normaliza el estado cuántico."""
        norma = np.sqrt(abs(self.alpha)**2 + abs(self.beta)**2)
        if norma > 0:
            self.alpha /= norma
            self.beta /= norma
    
    def probabilidad_cero(self) -> float:
        """Probabilidad de medir |0⟩."""
        return abs(self.alpha)**2
    
    def probabilidad_uno(self) -> float:
        """Probabilidad de medir |1⟩."""
        return abs(self.beta)**2
    
    def es_valido(self) -> bool:
        """Verifica si el estado es válido (normalizado)."""
        suma_prob = self.probabilidad_cero() + self.probabilidad_uno()
        return abs(suma_prob - 1.0) < 1e-10

@dataclass
class PulsoMicroondas:
    """Representa un pulso de microondas para control de qubits."""
    frecuencia: float  # Hz
    amplitud: float    # Voltios
    fase: float        # Radianes
    duracion: float    # Segundos
    forma_onda: str = "gaussian"  # gaussian, square, ramp
    
    def __post_init__(self):
        """Valida los parámetros del pulso."""
        if self.frecuencia <= 0:
            raise ValueError("La frecuencia debe ser positiva")
        if self.amplitud < 0:
            raise ValueError("La amplitud no puede ser negativa")
        if self.duracion <= 0:
            raise ValueError("La duración debe ser positiva")
        if self.fase < 0 or self.fase >= 2*np.pi:
            self.fase = self.fase % (2*np.pi)

@dataclass
class EstadoFoton:
    """Representa el estado de un fotón."""
    polarizacion: complex  # Polarización horizontal/vertical
    fase: float           # Fase óptica
    frecuencia: float     # Frecuencia del fotón
    intensidad: float     # Intensidad normalizada
    
    def __post_init__(self):
        """Valida el estado del fotón."""
        if self.intensidad < 0 or self.intensidad > 1:
            raise ValueError("La intensidad debe estar entre 0 y 1")
        if self.frecuencia <= 0:
            raise ValueError("La frecuencia debe ser positiva")

@dataclass
class MetricasSistema:
    """Métricas globales del sistema."""
    temperatura: float = 0.015  # Kelvin (temperatura de dilución)
    tiempo_coherencia_t1: float = 100e-6  # Segundos
    tiempo_coherencia_t2: float = 50e-6   # Segundos
    fidelidad_gates: float = 0.999
    eficiencia_conversion: float = 0.85
    ruido_ambiente: float = 0.01
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def actualizar_metricas(self, **kwargs):
        """Actualiza las métricas del sistema."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.timestamp = datetime.now(timezone.utc)

class TipoPulso(Enum):
    """Tipos de pulsos cuánticos soportados."""
    HADAMARD = "hadamard"
    PAULI_X = "pauli_x"
    PAULI_Y = "pauli_y"
    PAULI_Z = "pauli_z"
    ROTATION_X = "rotation_x"
    ROTATION_Y = "rotation_y"
    ROTATION_Z = "rotation_z"
    IDENTITY = "identity"

class QubitError(Exception):
    """Excepción base para errores relacionados con qubits."""
    pass

class DecoherenciaError(QubitError):
    """Error por decoherencia del qubit."""
    pass

class PulsoInvalidoError(QubitError):
    """Error por pulso inválido."""
    pass

class Qubit:
    """
    Simula un único qubit superconductor con decoherencia realista,
    manejo de errores robusto y monitoreo detallado.
    """
    
    def __init__(self, id_qubit: str, metricas: MetricasSistema):
        """
        Inicializa un qubit superconductor.
        
        Args:
            id_qubit: Identificador único del qubit
            metricas: Referencia a las métricas globales del sistema
        """
        if not id_qubit or not id_qubit.strip():
            raise ValueError("ID del qubit no puede estar vacío")
        
        self.id = id_qubit.strip()
        self.estado = EstadoComplejo()
        self.metricas = metricas
        
        # Historial y estadísticas
        self.historial_operaciones = []
        self.tiempo_ultima_operacion = time.time()
        self.numero_operaciones = 0
        self.errores_acumulados = 0
        
        # Estado de calibración
        self.calibrado = True
        self.ultima_calibracion = datetime.now(timezone.utc)
        
        # Lock para operaciones thread-safe
        self._lock = threading.Lock()
        
        logger.info(f"Qubit {self.id} inicializado correctamente")
    
    def _simular_decoherencia(self, tiempo_transcurrido: float) -> None:
        """
        Simula la decoherencia natural del qubit basada en T1 y T2.
        
        Args:
            tiempo_transcurrido: Tiempo en segundos desde la última operación
        """
        if tiempo_transcurrido <= 0:
            return
            
        # Decoherencia por relajación T1 (pérdida de energía)
        factor_t1 = np.exp(-tiempo_transcurrido / self.metricas.tiempo_coherencia_t1)
        
        # Decoherencia por defasaje T2 (pérdida de coherencia de fase)
        factor_t2 = np.exp(-tiempo_transcurrido / self.metricas.tiempo_coherencia_t2)
        
        # Aplicar decoherencia al estado
        if abs(self.estado.beta) > 0:  # Solo si hay superposición
            # Reducir amplitud del estado excitado
            self.estado.beta *= factor_t1
            
            # Agregar ruido de fase
            ruido_fase = np.random.normal(0, 1 - factor_t2)
            estado_plus = EstadoComplejo(1/np.sqrt(2) + 0j, 1/np.sqrt(2) + 0j)
            foton_plus = self.convertir(estado_plus)
            
            # Verificar que las conversiones mantienen coherencia
            intensidad_esperada_cero = 0.0  # |0⟩ no debería tener intensidad significativa
            intensidad_esperada_uno = 1.0   # |1⟩ debería tener máxima intensidad
            intensidad_esperada_plus = 0.5  # |+⟩ debería tener intensidad intermedia
            
            # Tolerancia para las verificaciones
            tolerancia = 0.2
            
            test_cero = abs(foton_cero.intensidad - intensidad_esperada_cero) < tolerancia
            test_uno = abs(foton_uno.intensidad - intensidad_esperada_uno) < tolerancia
            test_plus = abs(foton_plus.intensidad - intensidad_esperada_plus) < tolerancia
            
            calibracion_exitosa = test_cero and test_uno and test_plus
            
            if calibracion_exitosa:
                logger.info("Calibración del transductor completada exitosamente")
            else:
                logger.warning(f"Calibración falló - Tests: cero={test_cero}, uno={test_uno}, plus={test_plus}")
            
            return calibracion_exitosa
            
        except Exception as e:
            logger.error(f"Error durante calibración del transductor: {e}")
            return False

class GestorSistema:
    """
    Gestiona múltiples qubits y transductores, proporcionando
    una interfaz unificada para operaciones del sistema cuántico.
    """
    
    def __init__(self, num_qubits: int = 1):
        """
        Inicializa el gestor del sistema cuántico.
        
        Args:
            num_qubits: Número de qubits a crear
        """
        if num_qubits < 1:
            raise ValueError("Debe haber al menos un qubit")
        
        self.metricas = MetricasSistema()
        self.qubits = {}
        self.transductores = {}
        
        # Crear qubits
        for i in range(num_qubits):
            qubit_id = f"q{i}"
            self.qubits[qubit_id] = Qubit(qubit_id, self.metricas)
            
            # Crear transductor asociado
            transductor_id = f"t{i}"
            self.transductores[transductor_id] = Transductor(self.metricas)
        
        # Estadísticas globales
        self.tiempo_inicio = datetime.now(timezone.utc)
        self.operaciones_totales = 0
        
        # Lock para operaciones del sistema
        self._lock = threading.Lock()
        
        logger.info(f"Sistema cuántico inicializado con {num_qubits} qubits")
    
    def obtener_qubit(self, qubit_id: str) -> Qubit:
        """
        Obtiene un qubit por su ID.
        
        Args:
            qubit_id: ID del qubit
            
        Returns:
            Instancia del qubit
            
        Raises:
            KeyError: Si el qubit no existe
        """
        if qubit_id not in self.qubits:
            raise KeyError(f"Qubit {qubit_id} no encontrado")
        return self.qubits[qubit_id]
    
    def obtener_transductor(self, transductor_id: str) -> Transductor:
        """
        Obtiene un transductor por su ID.
        
        Args:
            transductor_id: ID del transductor
            
        Returns:
            Instancia del transductor
            
        Raises:
            KeyError: Si el transductor no existe
        """
        if transductor_id not in self.transductores:
            raise KeyError(f"Transductor {transductor_id} no encontrado")
        return self.transductores[transductor_id]
    
    def calibrar_sistema(self) -> Dict[str, bool]:
        """
        Calibra todos los qubits y transductores del sistema.
        
        Returns:
            Diccionario con resultados de calibración
        """
        resultados = {}
        
        with self._lock:
            logger.info("Iniciando calibración completa del sistema...")
            
            # Calibrar qubits
            for qubit_id, qubit in self.qubits.items():
                try:
                    resultado = qubit.calibrar()
                    resultados[f"qubit_{qubit_id}"] = resultado
                    if resultado:
                        logger.info(f"Qubit {qubit_id} calibrado exitosamente")
                    else:
                        logger.warning(f"Calibración del qubit {qubit_id} falló")
                except Exception as e:
                    logger.error(f"Error calibrando qubit {qubit_id}: {e}")
                    resultados[f"qubit_{qubit_id}"] = False
            
            # Calibrar transductores
            for trans_id, transductor in self.transductores.items():
                try:
                    resultado = transductor.calibrar_transductor()
                    resultados[f"transductor_{trans_id}"] = resultado
                    if resultado:
                        logger.info(f"Transductor {trans_id} calibrado exitosamente")
                    else:
                        logger.warning(f"Calibración del transductor {trans_id} falló")
                except Exception as e:
                    logger.error(f"Error calibrando transductor {trans_id}: {e}")
                    resultados[f"transductor_{trans_id}"] = False
            
            # Resumen de calibración
            total_componentes = len(resultados)
            componentes_exitosos = sum(resultados.values())
            tasa_exito = componentes_exitosos / total_componentes if total_componentes > 0 else 0
            
            logger.info(f"Calibración completada: {componentes_exitosos}/{total_componentes} componentes exitosos ({tasa_exito:.1%})")
            
            return resultados
    
    def ejecutar_circuito_cuantico(self, circuito: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ejecuta un circuito cuántico definido como lista de operaciones.
        
        Args:
            circuito: Lista de operaciones del circuito
            
        Returns:
            Resultados de la ejecución
        """
        resultados = {
            "mediciones": {},
            "estados_finales": {},
            "operaciones_ejecutadas": 0,
            "tiempo_ejecucion": 0,
            "errores": []
        }
        
        inicio_tiempo = time.time()
        
        try:
            with self._lock:
                for i, operacion in enumerate(circuito):
                    try:
                        tipo_op = operacion.get("tipo")
                        qubit_id = operacion.get("qubit")
                        
                        if not qubit_id or qubit_id not in self.qubits:
                            raise ValueError(f"Qubit inválido: {qubit_id}")
                        
                        qubit = self.qubits[qubit_id]
                        
                        if tipo_op == "gate":
                            gate_type = TipoPulso(operacion.get("gate"))
                            parametro = operacion.get("parametro", 0.0)
                            qubit.aplicar_gate_estandar(gate_type, parametro)
                            
                        elif tipo_op == "pulso":
                            pulso = PulsoMicroondas(
                                frecuencia=operacion.get("frecuencia"),
                                amplitud=operacion.get("amplitud"),
                                fase=operacion.get("fase"),
                                duracion=operacion.get("duracion"),
                                forma_onda=operacion.get("forma_onda", "gaussian")
                            )
                            qubit.aplicar_pulso(pulso)
                            
                        elif tipo_op == "medicion":
                            resultado = qubit.medir()
                            resultados["mediciones"][qubit_id] = resultado
                            
                        elif tipo_op == "reset":
                            qubit.reset()
                            
                        elif tipo_op == "conversion":
                            transductor_id = operacion.get("transductor", f"t{qubit_id[1:]}")
                            if transductor_id in self.transductores:
                                transductor = self.transductores[transductor_id]
                                estado_foton = transductor.convertir(qubit.estado)
                                resultados[f"foton_{qubit_id}"] = {
                                    "intensidad": estado_foton.intensidad,
                                    "fase": estado_foton.fase,
                                    "frecuencia": estado_foton.frecuencia
                                }
                            
                        else:
                            raise ValueError(f"Tipo de operación no soportado: {tipo_op}")
                        
                        resultados["operaciones_ejecutadas"] += 1
                        self.operaciones_totales += 1
                        
                    except Exception as e:
                        error_msg = f"Error en operación {i}: {e}"
                        resultados["errores"].append(error_msg)
                        logger.error(error_msg)
                
                # Capturar estados finales
                for qubit_id, qubit in self.qubits.items():
                    resultados["estados_finales"][qubit_id] = qubit.obtener_estado_info()
                
                resultados["tiempo_ejecucion"] = time.time() - inicio_tiempo
                
                logger.info(f"Circuito ejecutado: {resultados['operaciones_ejecutadas']} operaciones en {resultados['tiempo_ejecucion']:.3f}s")
                
                return resultados
                
        except Exception as e:
            resultados["errores"].append(f"Error crítico en ejecución: {e}")
            resultados["tiempo_ejecucion"] = time.time() - inicio_tiempo
            logger.error(f"Error crítico ejecutando circuito: {e}")
            return resultados
    
    def obtener_estado_sistema(self) -> Dict[str, Any]:
        """Retorna el estado completo del sistema."""
        with self._lock:
            estados_qubits = {}
            for qubit_id, qubit in self.qubits.items():
                estados_qubits[qubit_id] = qubit.obtener_estado_info()
            
            estadisticas_transductores = {}
            for trans_id, transductor in self.transductores.items():
                estadisticas_transductores[trans_id] = transductor.obtener_estadisticas()
            
            tiempo_funcionamiento = datetime.now(timezone.utc) - self.tiempo_inicio
            
            return {
                "metricas_sistema": {
                    "temperatura": self.metricas.temperatura,
                    "tiempo_coherencia_t1": self.metricas.tiempo_coherencia_t1,
                    "tiempo_coherencia_t2": self.metricas.tiempo_coherencia_t2,
                    "fidelidad_gates": self.metricas.fidelidad_gates,
                    "eficiencia_conversion": self.metricas.eficiencia_conversion,
                    "ruido_ambiente": self.metricas.ruido_ambiente,
                    "timestamp": self.metricas.timestamp.isoformat()
                },
                "qubits": estados_qubits,
                "transductores": estadisticas_transductores,
                "sistema": {
                    "tiempo_funcionamiento_segundos": tiempo_funcionamiento.total_seconds(),
                    "operaciones_totales": self.operaciones_totales,
                    "numero_qubits": len(self.qubits),
                    "numero_transductores": len(self.transductores)
                }
            }
    
    def actualizar_metricas_sistema(self, **kwargs):
        """Actualiza las métricas globales del sistema."""
        with self._lock:
            self.metricas.actualizar_metricas(**kwargs)
            logger.info(f"Métricas del sistema actualizadas: {kwargs}")
    
    def shutdown(self):
        """Apaga el sistema de forma segura."""
        with self._lock:
            logger.info("Iniciando apagado del sistema...")
            
            # Reset de todos los qubits
            for qubit_id, qubit in self.qubits.items():
                try:
                    qubit.reset()
                    logger.debug(f"Qubit {qubit_id} reiniciado")
                except Exception as e:
                    logger.error(f"Error reiniciando qubit {qubit_id}: {e}")
            
            # Limpiar historiales
            for qubit in self.qubits.values():
                qubit.historial_operaciones.clear()
            
            for transductor in self.transductores.values():
                transductor.historial_conversiones.clear()
            
            logger.info("Sistema apagado correctamente")

# Funciones de utilidad para facilitar el uso
def crear_sistema_basico(num_qubits: int = 1) -> GestorSistema:
    """
    Crea un sistema cuántico básico con configuración por defecto.
    
    Args:
        num_qubits: Número de qubits a crear
        
    Returns:
        Gestor del sistema inicializado
    """
    sistema = GestorSistema(num_qubits)
    
    # Calibración inicial
    resultados_calibracion = sistema.calibrar_sistema()
    
    # Verificar que al menos algunos componentes estén calibrados
    componentes_calibrados = sum(resultados_calibracion.values())
    if componentes_calibrados == 0:
        logger.warning("Ningún componente se calibró correctamente")
    else:
        logger.info(f"Sistema básico creado con {componentes_calibrados} componentes calibrados")
    
    return sistema

def ejemplo_bell_state(sistema: GestorSistema) -> Dict[str, Any]:
    """
    Ejemplo de creación de un estado de Bell usando el sistema.
    
    Args:
        sistema: Sistema cuántico a usar
        
    Returns:
        Resultados del experimento
    """
    if len(sistema.qubits) < 2:
        raise ValueError("Se necesitan al menos 2 qubits para crear un estado de Bell")
    
    # Circuito para estado de Bell |Φ+⟩ = (|00⟩ + |11⟩)/√2
    circuito_bell = [
        {"tipo": "reset", "qubit": "q0"},
        {"tipo": "reset", "qubit": "q1"},
        {"tipo": "gate", "qubit": "q0", "gate": "hadamard"},
        # Nota: CNOT requeriría implementación de gates de dos qubits
        # Por simplicidad, aplicamos X condicional simulado
        {"tipo": "gate", "qubit": "q1", "gate": "pauli_x"},
        {"tipo": "medicion", "qubit": "q0"},
        {"tipo": "medicion", "qubit": "q1"}
    ]
    
    return sistema.ejecutar_circuito_cuantico(circuito_bell)

# Ejemplo de uso
if __name__ == "__main__":
    # Crear sistema de prueba
    sistema = crear_sistema_basico(num_qubits=2)
    
    # Mostrar estado inicial
    estado_inicial = sistema.obtener_estado_sistema()
    print("Estado inicial del sistema:")
    print(f"Temperatura: {estado_inicial['metricas_sistema']['temperatura']} K")
    print(f"Número de qubits: {estado_inicial['sistema']['numero_qubits']}")
    
    # Ejecutar ejemplo de Bell state
    try:
        resultados_bell = ejemplo_bell_state(sistema)
        print("\nResultados del estado de Bell:")
        print(f"Mediciones: {resultados_bell['mediciones']}")
        print(f"Operaciones ejecutadas: {resultados_bell['operaciones_ejecutadas']}")
        print(f"Tiempo de ejecución: {resultados_bell['tiempo_ejecucion']:.3f}s")
        
        if resultados_bell['errores']:
            print(f"Errores encontrados: {resultados_bell['errores']}")
            
    except Exception as e:
        print(f"Error ejecutando ejemplo: {e}")
    
    # Apagar sistema
    sistema.shutdown()
    print("\nSistema apagado correctamente")
