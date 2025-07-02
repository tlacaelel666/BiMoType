# components/ai_engine.py
import logging
import re
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime, timezone

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TipoOperacion(Enum):
    """Tipos de operaciones soportadas por el motor de IA."""
    QUANTUM_SIMULATION = "quantum_simulation"
    ANOMALY_DETECTION = "anomaly_detection"
    TOPOLOGICAL_ANALYSIS = "topological_analysis"
    CLASSICAL_OPTIMIZATION = "classical_optimization"
    HYBRID_PROCESSING = "hybrid_processing"
    UNKNOWN = "unknown"

@dataclass
class ComandoAnalizado:
    """Estructura para comandos analizados."""
    comando_original: str
    tipo_operacion: TipoOperacion
    parametros: Dict[str, Any] = field(default_factory=dict)
    confianza: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadatos: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConfiguracionIA:
    """Configuración del motor de IA."""
    umbral_confianza: float = 0.7
    max_qubits: int = 100
    max_profundidad: int = 1000
    enable_logging: bool = True
    modelos_habilitados: List[str] = field(default_factory=lambda: [
        "mahalanobis", "vietoris_rips", "keras", "quantum_simulator"
    ])

class AIEngineError(Exception):
    """Excepción base para errores del motor de IA."""
    pass

class ComandoInvalidoError(AIEngineError):
    """Error para comandos inválidos o no reconocidos."""
    pass

class ParametrosInvalidosError(AIEngineError):
    """Error para parámetros fuera de rango o inválidos."""
    pass

class AIEngine:
    """
    Motor de IA robusto que interpreta comandos y genera cargas útiles
    para el protocolo BiMoType con validación exhaustiva y manejo de errores.
    """
    
    def __init__(self, configuracion: Optional[ConfiguracionIA] = None):
        """
        Inicializa el motor de IA con configuración personalizable.
        
        Args:
            configuracion: Configuración opcional del motor
        """
        self.config = configuracion or ConfiguracionIA()
        self.modelos_cargados = {}
        self.historial_comandos = []
        self.estadisticas = {
            "comandos_procesados": 0,
            "errores_encontrados": 0,
            "tiempo_promedio_procesamiento": 0.0
        }
        
        # Patrones de reconocimiento de comandos
        self.patrones_comando = {
            TipoOperacion.QUANTUM_SIMULATION: [
                r"simula.*qubit|quantum.*simulation|ejecutar.*circuito",
                r"hadamard|pauli|cnot|rotation.*gate"
            ],
            TipoOperacion.ANOMALY_DETECTION: [
                r"detectar.*anomalia|buscar.*outlier|analizar.*patron",
                r"mahalanobis|cluster.*analysis"
            ],
            TipoOperacion.TOPOLOGICAL_ANALYSIS: [
                r"topologia|vietoris.*rips|persistent.*homology",
                r"analizar.*estructura|mapper.*algorithm"
            ]
        }
        
        self._inicializar_modelos()
        
        if self.config.enable_logging:
            logger.info(f"AIEngine inicializado con {len(self.config.modelos_habilitados)} modelos")
    
    def _inicializar_modelos(self):
        """Inicializa los modelos de IA según la configuración."""
        try:
            for modelo in self.config.modelos_habilitados:
                if modelo == "mahalanobis":
                    self.modelos_cargados[modelo] = self._crear_modelo_mahalanobis()
                elif modelo == "vietoris_rips":
                    self.modelos_cargados[modelo] = self._crear_modelo_vietoris()
                elif modelo == "keras":
                    self.modelos_cargados[modelo] = self._crear_modelo_keras()
                elif modelo == "quantum_simulator":
                    self.modelos_cargados[modelo] = self._crear_simulador_cuantico()
                    
            logger.info(f"Modelos inicializados: {list(self.modelos_cargados.keys())}")
            
        except Exception as e:
            logger.error(f"Error inicializando modelos: {e}")
            raise AIEngineError(f"Fallo en inicialización de modelos: {e}")
    
    def _crear_modelo_mahalanobis(self) -> Dict[str, Any]:
        """Crea y configura el modelo de distancia de Mahalanobis."""
        return {
            "tipo": "anomaly_detector",
            "umbral": 2.5,
            "matriz_covarianza": None,
            "datos_entrenamiento": None
        }
    
    def _crear_modelo_vietoris(self) -> Dict[str, Any]:
        """Crea y configura el complejo de Vietoris-Rips."""
        return {
            "tipo": "topological_analyzer",
            "epsilon": 0.5,
            "max_dimension": 3,
            "filtro_persistencia": True
        }
    
    def _crear_modelo_keras(self) -> Dict[str, Any]:
        """Crea y configura el modelo de red neuronal."""
        return {
            "tipo": "neural_network",
            "arquitectura": "transformer",
            "capas": 12,
            "dimensiones": 768
        }
    
    def _crear_simulador_cuantico(self) -> Dict[str, Any]:
        """Crea y configura el simulador cuántico."""
        return {
            "tipo": "quantum_simulator",
            "backend": "statevector",
            "precision": "double",
            "max_entanglement": 1.0
        }
    
    def _clasificar_comando(self, comando: str) -> Tuple[TipoOperacion, float]:
        """
        Clasifica un comando usando patrones regex y análisis semántico.
        
        Args:
            comando: Comando de texto a clasificar
            
        Returns:
            Tupla con (tipo_operacion, confianza)
        """
        comando_lower = comando.lower()
        mejores_coincidencias = []
        
        for tipo_op, patrones in self.patrones_comando.items():
            coincidencias = 0
            for patron in patrones:
                if re.search(patron, comando_lower):
                    coincidencias += 1
            
            if coincidencias > 0:
                confianza = min(coincidencias / len(patrones), 1.0)
                mejores_coincidencias.append((tipo_op, confianza))
        
        if not mejores_coincidencias:
            return TipoOperacion.UNKNOWN, 0.0
        
        # Devolver el tipo con mayor confianza
        mejor_tipo, mejor_confianza = max(mejores_coincidencias, key=lambda x: x[1])
        return mejor_tipo, mejor_confianza
    
    def _extraer_parametros(self, comando: str, tipo_operacion: TipoOperacion) -> Dict[str, Any]:
        """
        Extrae parámetros específicos del comando según el tipo de operación.
        
        Args:
            comando: Comando original
            tipo_operacion: Tipo de operación identificado
            
        Returns:
            Diccionario con parámetros extraídos
        """
        parametros = {}
        
        try:
            if tipo_operacion == TipoOperacion.QUANTUM_SIMULATION:
                parametros.update(self._extraer_parametros_cuanticos(comando))
            elif tipo_operacion == TipoOperacion.ANOMALY_DETECTION:
                parametros.update(self._extraer_parametros_anomalia(comando))
            elif tipo_operacion == TipoOperacion.TOPOLOGICAL_ANALYSIS:
                parametros.update(self._extraer_parametros_topologia(comando))
                
        except Exception as e:
            logger.warning(f"Error extrayendo parámetros: {e}")
            
        return parametros
    
    def _extraer_parametros_cuanticos(self, comando: str) -> Dict[str, Any]:
        """Extrae parámetros específicos para simulación cuántica."""
        parametros = {
            "qubits": 2,  # Valor por defecto
            "profundidad": 5,
            "gates": [],
            "mediciones": True
        }
        
        # Buscar número de qubits
        qubit_match = re.search(r'(\d+)\s*qubit', comando.lower())
        if qubit_match:
            qubits = int(qubit_match.group(1))
            if qubits <= self.config.max_qubits:
                parametros["qubits"] = qubits
            else:
                raise ParametrosInvalidosError(f"Número de qubits ({qubits}) excede el máximo ({self.config.max_qubits})")
        
        # Buscar profundidad del circuito
        depth_match = re.search(r'profundidad\s*(\d+)|depth\s*(\d+)', comando.lower())
        if depth_match:
            profundidad = int(depth_match.group(1) or depth_match.group(2))
            if profundidad <= self.config.max_profundidad:
                parametros["profundidad"] = profundidad
            else:
                raise ParametrosInvalidosError(f"Profundidad ({profundidad}) excede el máximo ({self.config.max_profundidad})")
        
        # Identificar gates específicos
        gates_encontrados = []
        if re.search(r'hadamard|h\s*gate', comando.lower()):
            gates_encontrados.append("hadamard")
        if re.search(r'pauli|[xyz]\s*gate', comando.lower()):
            gates_encontrados.append("pauli")
        if re.search(r'cnot|cx\s*gate', comando.lower()):
            gates_encontrados.append("cnot")
            
        parametros["gates"] = gates_encontrados
        return parametros
    
    def _extraer_parametros_anomalia(self, comando: str) -> Dict[str, Any]:
        """Extrae parámetros para detección de anomalías."""
        return {
            "metodo": "mahalanobis",
            "umbral_sensibilidad": 0.05,
            "ventana_temporal": 100,
            "normalizacion": True
        }
    
    def _extraer_parametros_topologia(self, comando: str) -> Dict[str, Any]:
        """Extrae parámetros para análisis topológico."""
        return {
            "epsilon": 0.5,
            "max_dimension": 2,
            "filtro_persistencia": True,
            "metrica": "euclidiana"
        }
    
    def _generar_prediccion_ia(self, comando_analizado: ComandoAnalizado) -> Dict[str, Any]:
        """
        Genera predicciones usando los modelos cargados.
        
        Args:
            comando_analizado: Comando ya procesado y validado
            
        Returns:
            Diccionario con predicciones y métricas
        """
        prediccion = {
            "fidelidad_esperada": 0.85,
            "riesgo_error": 0.10,
            "tiempo_estimado": 0.0,
            "recursos_necesarios": {},
            "recomendaciones": []
        }
        
        try:
            if comando_analizado.tipo_operacion == TipoOperacion.QUANTUM_SIMULATION:
                # Calcular métricas específicas para simulación cuántica
                qubits = comando_analizado.parametros.get("qubits", 2)
                profundidad = comando_analizado.parametros.get("profundidad", 5)
                
                # Fidelidad decrece exponencialmente con la complejidad
                complejidad = qubits * profundidad
                prediccion["fidelidad_esperada"] = max(0.1, 0.95 * np.exp(-complejidad * 0.001))
                prediccion["riesgo_error"] = 1.0 - prediccion["fidelidad_esperada"]
                prediccion["tiempo_estimado"] = complejidad * 0.01  # segundos
                
                prediccion["recursos_necesarios"] = {
                    "memoria_ram": f"{2**qubits * 32}MB",
                    "cpu_cores": min(qubits, 8),
                    "tiempo_cpu": f"{prediccion['tiempo_estimado']:.2f}s"
                }
                
                if prediccion["fidelidad_esperada"] < 0.5:
                    prediccion["recomendaciones"].append("Considerar reducir la complejidad del circuito")
                
        except Exception as e:
            logger.error(f"Error generando predicción: {e}")
            prediccion["error"] = str(e)
            
        return prediccion
    
    def _validar_comando(self, comando: str) -> None:
        """
        Valida que el comando sea procesable.
        
        Args:
            comando: Comando a validar
            
        Raises:
            ComandoInvalidoError: Si el comando no es válido
        """
        if not comando or not comando.strip():
            raise ComandoInvalidoError("Comando vacío o solo espacios en blanco")
        
        if len(comando) > 1000:
            raise ComandoInvalidoError("Comando excede la longitud máxima (1000 caracteres)")
        
        # Verificar caracteres peligrosos
        caracteres_peligrosos = ['<script>', '${', '`', 'eval(']
        for char in caracteres_peligrosos:
            if char in comando.lower():
                raise ComandoInvalidoError(f"Comando contiene caracteres no permitidos: {char}")
    
    def analizar_comando(self, comando_usuario: str) -> Dict[str, Any]:
        """
        Procesa un comando de texto y lo convierte en una carga útil
        rica en metadatos para la transmisión.
        
        Args:
            comando_usuario: Comando de texto del usuario
            
        Returns:
            Diccionario con la carga útil generada
            
        Raises:
            ComandoInvalidoError: Si el comando no es válido
            ParametrosInvalidosError: Si los parámetros están fuera de rango
            AIEngineError: Para otros errores del motor
        """
        inicio_tiempo = datetime.now()
        
        try:
            # 1. Validar comando
            self._validar_comando(comando_usuario)
            
            if self.config.enable_logging:
                logger.info(f"Analizando comando: {comando_usuario[:50]}...")
            
            # 2. Clasificar comando y extraer tipo
            tipo_operacion, confianza = self._clasificar_comando(comando_usuario)
            
            if confianza < self.config.umbral_confianza:
                logger.warning(f"Baja confianza en clasificación: {confianza:.2f}")
            
            # 3. Extraer parámetros específicos
            parametros = self._extraer_parametros(comando_usuario, tipo_operacion)
            
            # 4. Crear comando analizado
            comando_analizado = ComandoAnalizado(
                comando_original=comando_usuario,
                tipo_operacion=tipo_operacion,
                parametros=parametros,
                confianza=confianza,
                metadatos={
                    "longitud_comando": len(comando_usuario),
                    "palabras_clave_detectadas": len([p for patrones in self.patrones_comando.get(tipo_operacion, []) 
                                                     for p in patrones if re.search(p, comando_usuario.lower())]),
                    "modelos_disponibles": list(self.modelos_cargados.keys())
                }
            )
            
            # 5. Generar predicción con IA
            prediccion_ia = self._generar_prediccion_ia(comando_analizado)
            
            # 6. Ensamblar carga útil final
            carga_util_generada = {
                "comando_original": comando_usuario,
                "tipo_operacion": tipo_operacion.value,
                "parametros_extraidos": parametros,
                "confianza_clasificacion": confianza,
                "prediccion_ia": prediccion_ia,
                "timestamp": comando_analizado.timestamp.isoformat(),
                "metadatos": comando_analizado.metadatos,
                "firma_radiactiva": self._generar_firma_radiactiva(comando_analizado),
                "version_protocolo": "BiMoType_v2.1",
                "id_sesion": hash(comando_usuario + str(inicio_tiempo)) % (10**8)
            }
            
            # 7. Actualizar estadísticas
            tiempo_procesamiento = (datetime.now() - inicio_tiempo).total_seconds()
            self._actualizar_estadisticas(tiempo_procesamiento, exito=True)
            
            # 8. Guardar en historial
            self.historial_comandos.append(comando_analizado)
            if len(self.historial_comandos) > 100:  # Mantener solo los últimos 100
                self.historial_comandos.pop(0)
            
            if self.config.enable_logging:
                logger.info(f"Comando procesado exitosamente en {tiempo_procesamiento:.3f}s")
            
            return carga_util_generada
            
        except (ComandoInvalidoError, ParametrosInvalidosError) as e:
            self._actualizar_estadisticas(0, exito=False)
            logger.error(f"Error de validación: {e}")
            raise
            
        except Exception as e:
            self._actualizar_estadisticas(0, exito=False)
            logger.error(f"Error inesperado procesando comando: {e}")
            raise AIEngineError(f"Error interno del motor de IA: {e}")
    
    def _generar_firma_radiactiva(self, comando: ComandoAnalizado) -> str:
        """
        Genera una firma única basada en el comando y sus características.
        
        Args:
            comando: Comando analizado
            
        Returns:
            Firma radiactiva como string hexadecimal
        """
        # Combinar elementos únicos del comando
        elementos = [
            comando.comando_original,
            comando.tipo_operacion.value,
            str(comando.confianza),
            str(comando.timestamp.timestamp())
        ]
        
        # Generar hash único
        contenido = ''.join(elementos)
        firma = hash(contenido) % (16**8)  # 8 dígitos hexadecimales
        
        return f"RAD_{firma:08X}"
    
    def _actualizar_estadisticas(self, tiempo_procesamiento: float, exito: bool):
        """Actualiza las estadísticas internas del motor."""
        self.estadisticas["comandos_procesados"] += 1
        
        if not exito:
            self.estadisticas["errores_encontrados"] += 1
        
        # Actualizar tiempo promedio (media móvil simple)
        if tiempo_procesamiento > 0:
            actual = self.estadisticas["tiempo_promedio_procesamiento"]
            total = self.estadisticas["comandos_procesados"]
            self.estadisticas["tiempo_promedio_procesamiento"] = (actual * (total - 1) + tiempo_procesamiento) / total
    
    def obtener_estadisticas(self) -> Dict[str, Any]:
        """Retorna las estadísticas actuales del motor."""
        return {
            **self.estadisticas,
            "tasa_exito": (self.estadisticas["comandos_procesados"] - self.estadisticas["errores_encontrados"]) / 
                         max(self.estadisticas["comandos_procesados"], 1),
            "modelos_cargados": len(self.modelos_cargados),
            "historial_size": len(self.historial_comandos)
        }
    
    def reiniciar_estadisticas(self):
        """Reinicia las estadísticas del motor."""
        self.estadisticas = {
            "comandos_procesados": 0,
            "errores_encontrados": 0,
            "tiempo_promedio_procesamiento": 0.0
        }
        self.historial_comandos.clear()
        logger.info("Estadísticas del motor reiniciadas")

# Ejemplo de uso
if __name__ == "__main__":
    # Configuración personalizada
    config = ConfiguracionIA(
        umbral_confianza=0.6,
        max_qubits=50,
        max_profundidad=100,
        enable_logging=True
    )
    
    # Crear motor de IA
    motor = AIEngine(config)
    
    # Comandos de prueba
    comandos_prueba = [
        "Simular un circuito cuántico con 4 qubits y profundidad 10",
        "Detectar anomalías usando distancia de Mahalanobis",
        "Analizar topología con Vietoris-Rips",
        "Comando inválido con <script>alert('hack')</script>",
        ""
    ]
    
    for comando in comandos_prueba:
        try:
            resultado = motor.analizar_comando(comando)
            print(f"\n✅ Comando: {comando}")
            print(f"Tipo: {resultado['tipo_operacion']}")
            print(f"Confianza: {resultado['confianza_clasificacion']:.2f}")
            print(f"Firma: {resultado['firma_radiactiva']}")
            
        except Exception as e:
            print(f"\n❌ Error con comando '{comando}': {e}")
    
    # Mostrar estadísticas finales
    print(f"\n📊 Estadísticas finales:")
    stats = motor.obtener_estadisticas()
    for key, value in stats.items():
        print(f"  {key}: {value}")
