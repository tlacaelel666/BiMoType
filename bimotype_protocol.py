# bimotype_v2/protocols/bimotype_protocol.py

import json
import time
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Importaciones relativas para las dataclasses y enums
from datatypes import (
    EstadoQubitConceptual, OperacionCuantica, TipoDecaimiento,
    EstadoComplejo, EstadoFoton, FirmaRadiactiva, PulsoMicroondas,
    QuantumRadiationState, PaqueteBiMoType, MetricasSistema
)

# Definición de configuraciones fuera de la clase principal para modularidad
# O como atributos de clase si son constantes para todas las instancias
RADIOACTIVE_ELEMENTS_CONFIG = {
    'U235': {'half_life_years': 7.038e8, 'energy': 202.5, 'type': TipoDecaimiento.FISSION, 'spin': 7/2},
    'U238': {'half_life_years': 4.468e9, 'energy': 4.27, 'type': TipoDecaimiento.ALPHA, 'spin': 0},
    'Pu239': {'half_life_years': 2.411e4, 'energy': 200.0, 'type': TipoDecaimiento.FISSION, 'spin': 1/2},
    'Pu238': {'half_life_years': 87.7, 'energy': 5.59, 'type': TipoDecaimiento.ALPHA, 'spin': 0},
    'Th232': {'half_life_years': 1.405e10, 'energy': 4.08, 'type': TipoDecaimiento.ALPHA, 'spin': 0},
    'Sr90': {'half_life_years': 28.8, 'energy': 0.546, 'type': TipoDecaimiento.BETA, 'spin': 0},
    'Co60': {'half_life_years': 5.27, 'energy': 2.82, 'type': TipoDecaimiento.BETA_GAMMA, 'spin': 5},
    'Cm244': {'half_life_years': 18.1, 'energy': 5.81, 'type': TipoDecaimiento.ALPHA, 'spin': 0},
    'Po210': {'half_life_years': 0.38, 'energy': 5.41, 'type': TipoDecaimiento.ALPHA, 'spin': 0},
    'Am241': {'half_life_years': 432.6, 'energy': 5.49, 'type': TipoDecaimiento.ALPHA, 'spin': 5/2},
    'Cf252': {'half_life_years': 2.65, 'energy': 6.12, 'type': TipoDecaimiento.ALPHA_SF, 'spin': 0},
    'Tc99m': {'half_life_years': 0.25, 'energy': 0.14, 'type': TipoDecaimiento.GAMMA, 'spin': 9/2}
}

MORSE_QUANTUM_MAP_CONFIG = {
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
    ' ': {'morse': '/', 'quantum': '⊗', 'isotope': 'vacuum', 'phase': 0}, # Espacio entre palabras
    '.': {'morse': '.-.-.-', 'quantum': '...', 'isotope': 'Tc99m', 'phase': 0}, # Punto
    ',': {'morse': '--..--', 'quantum': ',,,', 'isotope': 'Am241', 'phase': np.pi/4}, # Coma
    '?': {'morse': '..-.--', 'quantum': '???', 'isotope': 'Pu238', 'phase': np.pi/2}, # Signo de interrogación
    '!': {'morse': '-.-.--', 'quantum': '!!!', 'isotope': 'U235', 'phase': 3*np.pi/4}, # Signo de exclamación
}


class BiMoTypeProtocol: # Renombrado de SistemaQuantumBiMoType
    """
    Protocolo unificado para comunicación cuántica-radiactiva BiMOtype.
    Integra la codificación/decodificación de mensajes, simulación de estados cuánticos
    y monitoreo de métricas del sistema.
    """

    def __init__(self):
        self.radioactive_elements = RADIOACTIVE_ELEMENTS_CONFIG
        self.morse_quantum_map = MORSE_QUANTUM_MAP_CONFIG

        # Inicializar métricas del sistema
        self.metricas = MetricasSistema() # Usamos la dataclass MetricasSistema

    def crear_firma_radiactiva_basada_en_carga_util(self, payload: Dict[str, Any]) -> FirmaRadiactiva:
        """
        Genera una FirmaRadiactiva basada en el contenido de la carga útil,
        simulando la lógica del cuadrante-coremind (Mahalanobis, polaridades, etc.).
        """
        # Aquí es donde se integrarían las ecuaciones y lógica compleja del cuadrante-coremind
        # para derivar la firma de la carga útil.
        # Por ahora, usamos valores simulados o derivados heurísticamente.

        # Ejemplo de selección de isótopo y decaimiento basado en el mensaje
        message_length = len(json.dumps(payload))
        isotopo_base = 'U235' # Default
        tipo_decaimiento_base = TipoDecaimiento.FISSION
        energia_pico = 202.5
        vida_media = RADIOACTIVE_ELEMENTS_CONFIG['U235']['half_life_years'] * 3.154e7 # A segundos
        spin_nuclear = RADIOACTIVE_ELEMENTS_CONFIG['U235']['spin']

        if message_length > 50:
            isotopo_base = 'Pu239'
            tipo_decaimiento_base = TipoDecaimiento.FISSION
            energia_pico = 200.0
            vida_media = RADIOACTIVE_ELEMENTS_CONFIG['Pu239']['half_life_years'] * 3.154e7
            spin_nuclear = RADIOACTIVE_ELEMENTS_CONFIG['Pu239']['spin']
        elif "quantum" in json.dumps(payload).lower():
            isotopo_base = 'Sr90'
            tipo_decaimiento_base = TipoDecaimiento.BETA
            energia_pico = 0.546
            vida_media = RADIOACTIVE_ELEMENTS_CONFIG['Sr90']['half_life_years'] * 3.154e7
            spin_nuclear = RADIOACTIVE_ELEMENTS_CONFIG['Sr90']['spin']

        # Simulación de métricas del cuadrante-coremind
        # Estas se derivarían de los datos de la IA y el contexto
        mahalanobis_distance = np.random.rand() * 2.0 # Ejemplo: 0.0 a 2.0
        lambda_double_non_locality = np.random.rand() * 0.5 # Ejemplo: 0.0 a 0.5
        mg_polarity = np.random.rand() # Ejemplo: 0.0 a 1.0
        mg_threshold = 0.5 # Umbral fijo de ejemplo
        vacuum_polarity_n_r = np.random.rand() * 0.1 # Ejemplo: 0.0 a 0.1

        return FirmaRadiactiva(
            isotopo=isotopo_base,
            energia_pico_ev=energia_pico,
            tipo_decaimiento=tipo_decaimiento_base,
            vida_media_s=vida_media,
            spin_nuclear=spin_nuclear,
            mahalanobis_distance=mahalanobis_distance,
            lambda_double_non_locality=lambda_double_non_locality,
            mg_polarity=mg_polarity,
            mg_threshold=mg_threshold,
            vacuum_polarity_n_r=vacuum_polarity_n_r
        )

    def crear_estado_cuantico(self, char_data: Dict, firma_radiactiva_dominante: FirmaRadiactiva) -> QuantumRadiationState:
        """
        Crea un estado cuántico-radiactivo unificado a partir de datos de carácter
        y la firma radiactiva dominante.
        """
        phase = char_data['phase']
        
        # El estado de spin del fotón/qubit se inicializa con la fase del mapeo
        # Aquí se podría integrar la influencia de la firma radiactiva en el estado inicial
        # Por ejemplo, una ligera perturbación en alfa/beta basada en Mahalanobis
        alpha_init = np.cos(phase/2)
        beta_init = np.sin(phase/2)

        # Simular una ligera influencia de la firma en el estado cuántico inicial
        if firma_radiactiva_dominante.mahalanobis_distance is not None:
            perturbation = firma_radiactiva_dominante.mahalanobis_distance * 0.01 # Pequeña perturbación
            alpha_init += np.random.normal(0, perturbation)
            beta_init += np.random.normal(0, perturbation)

        spin_state = EstadoComplejo(
            alpha=complex(alpha_init, 0),
            beta=complex(beta_init, 0)
        )
        
        isotope = char_data['isotope']
        isotope_data = self.radioactive_elements.get(isotope, {})
        
        # Asegurarse de que la vida_media_s se obtiene de los datos del isótopo, no de la firma dominante
        life_s = isotope_data.get('half_life_years', 1.0) * 3.154e7
        decay_rate_val = 1.0 / life_s if life_s > 0 else 0.0

        return QuantumRadiationState(
            isotope=isotope,
            energy_level=isotope_data.get('energy', 1.0),
            decay_rate=decay_rate_val,
            spin_state=spin_state,
            entanglement_phase=phase, # La fase de entrelazamiento inicial
            coherence_time=self.calculate_coherence_time(isotope),
            firma_radiactiva=firma_radiactiva_dominante # Cada QRS lleva la firma del paquete
        )

    def encode_quantum_message(self, message: str) -> PaqueteBiMoType:
        """Codifica un mensaje en un paquete BiMOtype unificado."""
        timestamp = time.time()
        # Generar la firma radiactiva principal del paquete basada en el mensaje original
        firma_radiactiva_paquete = self.crear_firma_radiactiva_basada_en_carga_util({"mensaje": message})

        estados_cuanticos = []
        quantum_states_data = []

        for i, char in enumerate(message.upper()):
            if char in self.morse_quantum_map:
                char_data = self.morse_quantum_map[char]
                
                if char_data['isotope'] == 'vacuum': # Saltar caracteres que mapean al vacío
                    continue

                # Crear cada QuantumRadiationState, pasándole la firma_radiactiva_paquete
                quantum_state = self.crear_estado_cuantico(char_data, firma_radiactiva_paquete)
                estados_cuanticos.append(quantum_state)

                quantum_states_data.append({
                    'position': i,
                    'character': char,
                    'morse': char_data['morse'],
                    'quantum_state_representation': char_data['quantum'], # Representación string
                    'isotope': quantum_state.isotope,
                    'phase': quantum_state.entanglement_phase,
                    'estado_complejo_alpha': float(quantum_state.spin_state.alpha.real), # Solo la parte real para JSON simple
                    'estado_complejo_beta': float(quantum_state.spin_state.beta.real), # Solo la parte real para JSON simple
                    'energia': quantum_state.energy_level
                })
            else:
                # Manejo de caracteres no mapeados (opcional)
                print(f"Advertencia: Carácter '{char}' no encontrado en morse_quantum_map.")


        # Carga útil del paquete, incluyendo los datos de los estados cuánticos generados
        carga_util = {
            "protocolo": "BiMOtype-Quantum-Unified-v4.0",
            "mensaje_original": message,
            "estados_cuanticos_info": quantum_states_data, # Renombrado para evitar confusión
            "estadisticas_codificacion": {
                "total_qrs_generados": len(estados_cuanticos),
                "energia_total_qrs": sum(e.energy_level for e in estados_cuanticos),
                "tiempo_coherencia_promedio_qrs": np.mean([e.coherence_time for e in estados_cuanticos]) if estados_cuanticos else 0
            }
        }

        return PaqueteBiMoType(
            id_mensaje=f"BiMO-{int(timestamp)}-{abs(hash(message)) % 10000:04d}", # abs() para evitar hashes negativos
            timestamp=timestamp,
            carga_util=carga_util,
            firma_radiactiva=firma_radiactiva_paquete, # La firma general del paquete
            estados_cuanticos=estados_cuanticos # Los objetos QRS completos
        )

    def aplicar_operacion_cuantica(self, estado: EstadoComplejo, operacion: OperacionCuantica,
                                  parametros: Dict = None) -> EstadoComplejo:
        """Aplica operaciones cuánticas a un estado."""
        if parametros is None:
            parametros = {}

        if operacion == OperacionCuantica.HADAMARD:
            nueva_alpha = (estado.alpha + estado.beta) / np.sqrt(2)
            nueva_beta = (estado.alpha - estado.beta) / np.sqrt(2)
            return EstadoComplejo(nueva_alpha, nueva_beta)

        elif operacion == OperacionCuantica.ROTACION_X:
            angulo = parametros.get('angulo', np.pi/2)
            cos_half = np.cos(angulo/2)
            sin_half = 1j * np.sin(angulo/2)
            # Matrices de Pauli para rotaciones: RX = cos(theta/2)I - i sin(theta/2)sigma_x
            # sigma_x = [[0, 1], [1, 0]]
            # Multiplicar por el vector [alpha, beta]
            nueva_alpha = cos_half * estado.alpha - sin_half * estado.beta * 1j # Se corrije el i
            nueva_beta = -sin_half * estado.alpha * 1j + cos_half * estado.beta # Se corrije el i
            # Original: nueva_alpha = cos_half * estado.alpha + sin_half * estado.beta
            # Original: nueva_beta = sin_half * estado.alpha + cos_half * estado.beta
            # Corrected:
            nueva_alpha = cos_half * estado.alpha - 1j * np.sin(angulo/2) * estado.beta
            nueva_beta = -1j * np.sin(angulo/2) * estado.alpha + cos_half * estado.beta
            return EstadoComplejo(nueva_alpha, nueva_beta)


        elif operacion == OperacionCuantica.ROTACION_Y:
            angulo = parametros.get('angulo', np.pi/2)
            # RY = cos(theta/2)I + i sin(theta/2)sigma_y
            # sigma_y = [[0, -i], [i, 0]]
            cos_half = np.cos(angulo/2)
            sin_half = np.sin(angulo/2)
            nueva_alpha = cos_half * estado.alpha + sin_half * estado.beta
            nueva_beta = -sin_half * estado.alpha + cos_half * estado.beta
            return EstadoComplejo(nueva_alpha, nueva_beta)

        elif operacion == OperacionCuantica.ROTACION_Z:
            angulo = parametros.get('angulo', np.pi/2)
            # RZ = e^(-i theta/2 sigma_z)
            # sigma_z = [[1, 0], [0, -1]]
            exp_pos = np.exp(-1j * angulo/2)
            exp_neg = np.exp(1j * angulo/2)
            nueva_alpha = estado.alpha * exp_pos
            nueva_beta = estado.beta * exp_neg
            return EstadoComplejo(nueva_alpha, nueva_beta)

        elif operacion == OperacionCuantica.RESET:
            return EstadoComplejo(1.0, 0.0) # Estado |0⟩

        else:
            return estado # Retorna el estado sin cambios si la operación no es reconocida

    def simular_decoherencia(self, estado: EstadoComplejo, tiempo_ns: float) -> EstadoComplejo:
        """
        Simula efectos de decoherencia (T2 para decoherencia de fase, T1 para relajación de amplitud)
        en un estado cuántico.
        """
        # Convertir tiempos de microsegundos a nanosegundos
        T2_ns = self.metricas.decoherencia_t2_us * 1000
        T1_ns = self.metricas.decoherencia_t1_us * 1000 # Nuevo: tiempo de relajación

        # Factor de decoherencia de fase (T2)
        # Esto reduce la componente fuera de diagonal de la matriz de densidad (coherencia)
        if T2_ns > 0:
            decoherence_factor_t2 = np.exp(-tiempo_ns / T2_ns)
        else:
            decoherence_factor_t2 = 0.0 # Decoherencia instantánea

        # Factor de relajación de amplitud (T1)
        # Esto hace que el qubit decaiga hacia el estado de menor energía (|0⟩)
        if T1_ns > 0:
            prob_decay_to_0 = 1 - np.exp(-tiempo_ns / T1_ns)
        else:
            prob_decay_to_0 = 1.0 # Relajación instantánea

        # Aplicar T2: Reduce la coherencia (componente no diagonal)
        # Un enfoque simple es escalar las componentes no diagonales o las fases
        # Aquí, lo aplicamos a la fase de beta para simular la pérdida de coherencia
        # Considerar una matriz de densidad para una simulación más precisa de T1/T2
        # Por ahora, una simplificación:
        new_alpha = estado.alpha
        new_beta = estado.beta * decoherence_factor_t2 # Reduce la coherencia

        # Aplicar T1: Relajación de amplitud (probabilidad de decaer a |0>)
        # Esto es más complejo que una simple multiplicación. Podría ser un proceso estocástico.
        # Simplificación: si decae, el estado colapsa a |0> con probabilidad prob_decay_to_0
        if np.random.rand() < prob_decay_to_0:
            return EstadoComplejo(1.0, 0.0) # Colapsa a |0>

        # Añadir ruido térmico (generalmente afecta las amplitudes y fases)
        # El ruido se escala con la temperatura y un factor pequeño
        ruido_termico_mag = np.sqrt(self.metricas.temperatura_mk / 1000.0) * 0.01
        ruido_alpha = np.random.normal(0, ruido_termico_mag) + 1j * np.random.normal(0, ruido_termico_mag)
        ruido_beta = np.random.normal(0, ruido_termico_mag) + 1j * np.random.normal(0, ruido_termico_mag)

        new_alpha += ruido_alpha
        new_beta += ruido_beta

        return EstadoComplejo(new_alpha, new_beta)


    def decode_quantum_transmission(self, paquete: PaqueteBiMoType,
                                  mediciones_ruidosas: List[Dict]) -> Dict[str, Any]:
        """Decodifica una transmisión cuántica con métricas avanzadas."""
        try:
            decoded_chars = []
            fidelidades = []
            errores = []
            num_estados_decodificados = 0

            # Iterar sobre las mediciones ruidosas
            for i, medicion in enumerate(mediciones_ruidosas):
                if i >= len(paquete.estados_cuanticos):
                    errores.append(f"Medición extra en posición {i}.")
                    continue

                # Reconstruir estado cuántico a partir de la medición
                estado_reconstruido = self.reconstruir_estado_cuantico(medicion)

                # Simular decoherencia para el estado reconstruido (o aplicado antes de la medición)
                # Aplicamos la decoherencia al estado ideal *antes* de la comparación,
                # o asumimos que la medición ya refleja la decoherencia.
                # Aquí, simulamos un tiempo de vuelo de 50ns para cada 'qubit' de radiación
                estado_reconstruido_decohered = self.simular_decoherencia(estado_reconstruido, 50.0)

                # Encontrar la mejor coincidencia para el estado decohered
                # El estado ideal esperado (sin ruido ni decoherencia) proviene del paquete
                # Esto es crucial para la falseabilidad: ¿El estado medido coincide con el estado *esperado*?
                mejor_coincidencia, mejor_fidelidad = self.encontrar_mejor_coincidencia(
                    estado_reconstruido_decohered,
                    paquete.firma_radiactiva # Pasamos la firma para el filtrado en encontrar_mejor_coincidencia
                )

                if mejor_fidelidad > 0.7:
                    decoded_chars.append(mejor_coincidencia)
                    fidelidades.append(mejor_fidelidad)
                    num_estados_decodificados += 1
                else:
                    decoded_chars.append('?') # Carácter desconocido o con error
                    fidelidades.append(0.0)
                    errores.append(f"Posición {i}: Baja fidelidad ({mejor_fidelidad:.2f}) para carácter '{mejor_coincidencia}'.")


            fidelidad_promedio = np.mean(fidelidades) if fidelidades else 0.0
            self.metricas.fidelidad_promedio = fidelidad_promedio

            # Actualizar QBER (cuántos errores / total de mediciones esperadas)
            # Errores aquí se definen como fidelidad < 0.9.
            errores_cuanticos_qber = sum(1 for f in fidelidades if f < 0.9)
            self.metricas.qber_estimado = errores_cuanticos_qber / len(mediciones_ruidosas) if mediciones_ruidosas else 1.0

            # Actualizar métricas del sistema para el siguiente ciclo
            self.actualizar_metricas_sistema()

            return {
                "estado": "EXITO" if fidelidad_promedio > 0.8 else "DEGRADADO",
                "mensaje_decodificado": ''.join(decoded_chars),
                "metricas_cuanticas": {
                    "fidelidad_promedio": fidelidad_promedio,
                    "qber": self.metricas.qber_estimado,
                    "decoherencia_detectada": any(f < 0.5 for f in fidelidades), # Si alguna fidelidad cae mucho
                    "estados_decodificados": num_estados_decodificados
                },
                "errores": errores,
                "calidad_reconstruccion": self.evaluar_calidad(fidelidad_promedio),
                "metricas_sistema_finales": self.metricas # Retornar el objeto MetricasSistema
            }
        except Exception as e:
            # Captura cualquier excepción inesperada durante la decodificación
            print(f"Error crítico en decode_quantum_transmission: {e}")
            return {"estado": "ERROR", "mensaje_error": str(e), "metricas_sistema_finales": self.metricas}

    def calculate_coherence_time(self, isotope: str) -> float:
        """Calcula el tiempo de coherencia (simplificado) basado en la vida media del isótopo."""
        # Se asume T2 es proporcional a algún factor de la vida media
        # Esta es una simplificación; la coherencia T2 real es muy dependiente del entorno.
        # Aquí, se intenta asegurar un valor pequeño pero no cero.
        half_life_years = self.radioactive_elements.get(isotope, {}).get('half_life_years', 1.0)
        half_life_seconds = half_life_years * 3.154e7
        # Una forma común de relacionar es T2 ~ T1/2 o inversamente proporcional a la tasa de decaimiento
        # Aquí una heurística: cuanto más rápido decae, menor coherencia.
        # Limitar a un rango razonable para simulación (e.g., microsegundos)
        return max(1e-6, min(1e-3, 1.0 / (half_life_seconds + 1e-9))) # En segundos, asegura no division por cero

    def reconstruir_estado_cuantico(self, medicion: Dict) -> EstadoComplejo:
        """
        Reconstruye un EstadoComplejo a partir de una medición simulada.
        La 'energía_medida' se mapea a una fase, que luego define el estado del qubit.
        Esto simula la inferencia del estado a partir de los datos brutos del detector.
        """
        # La 'energía_medida' es una simplificación de un resultado de medición.
        # En la realidad, las mediciones cuánticas son probabilísticas (e.g., 0 o 1).
        # Para reconstruir un 'EstadoComplejo' a partir de mediciones, se necesitarían
        # múltiples ejecuciones del mismo circuito para obtener las probabilidades (alpha, beta)
        # o técnicas como la tomografía de estado cuántico.
        # Aquí, usaremos la energía para inferir una fase y construir un estado simple.
        
        # Mapeamos la energía a un ángulo de rotación para alpha y beta
        # Aseguramos que la fase esté en el rango [0, 2*pi]
        fase_estimada_norm = (medicion.get('energia_medida', 0) % 1.0) * 2 * np.pi
        
        # Creamos un estado en el plano XY (coherente)
        alpha_reconstructed = np.cos(fase_estimada_norm / 2.0) + 0.0j
        beta_reconstructed = np.sin(fase_estimada_norm / 2.0) + 0.0j
        
        # Podríamos añadir una ligera perturbación basada en la potencia de ruido del sistema
        ruido_magnitud = self.metricas.potencia_ruido_dbm / -100.0 * 0.01 # Escala el ruido
        alpha_reconstructed += np.random.normal(0, ruido_magnitud) + 1j * np.random.normal(0, ruido_magnitud)
        beta_reconstructed += np.random.normal(0, ruido_magnitud) + 1j * np.random.normal(0, ruido_magnitud)

        return EstadoComplejo(alpha_reconstructed, beta_reconstructed)


    def encontrar_mejor_coincidencia(self, estado_reconstruido: EstadoComplejo, firma_contexto: FirmaRadiactiva) -> Tuple[str, float]:
        """
        Encuentra la mejor coincidencia para un estado cuántico reconstruido
        comparándolo con los estados ideales definidos en el morse_quantum_map.
        Puede usar la 'firma_contexto' para refinar la búsqueda o ponderar.
        """
        mejor_char = '?'
        mejor_fidelidad = 0.0

        for char, char_data in self.morse_quantum_map.items():
            # Saltar el espacio que no se traduce a un QRS directo
            if char == ' ':
                continue

            # Si la firma del paquete indica un isótopo específico, podemos priorizar o filtrar
            # Por ejemplo, si el isotopo de la firma no es el mismo que el del char_data,
            # podríamos penalizar la fidelidad o ignorar la coincidencia.
            if firma_contexto.isotopo != char_data['isotope'] and char_data['isotope'] != 'vacuum':
                # Esto es una heurística para la falseabilidad: ¿coincide el "canal" de radiación?
                # Si no coincide el isótopo, la probabilidad de que sea una coincidencia es baja.
                continue

            # Crear estado ideal (sin ruido ni decoherencia) para comparación
            fase_ideal = char_data['phase']
            estado_ideal = EstadoComplejo(
                complex(np.cos(fase_ideal/2), 0),
                complex(np.sin(fase_ideal/2), 0)
            )

            # Calcular fidelidad cuántica entre el estado reconstruido y el ideal
            fidelidad = self.calcular_fidelidad_cuantica(estado_reconstruido, estado_ideal)

            # Considerar también el impacto de la polaridad MG si es relevante para la fidelidad
            if firma_contexto.mg_polarity is not None:
                # Ejemplo: si la polaridad es baja, la fidelidad percibida es menor
                fidelidad *= (1 - (1 - firma_contexto.mg_polarity) * 0.1) # Reduce fidelidad un 10% si polaridad es 0

            if fidelidad > mejor_fidelidad:
                mejor_fidelidad = fidelidad
                mejor_char = char

        return mejor_char, mejor_fidelidad

    def calcular_fidelidad_cuantica(self, estado1: EstadoComplejo, estado2: EstadoComplejo) -> float:
        """
        Calcula la fidelidad cuántica entre dos estados complejos normalizados.
        Fidelidad = |⟨ψ₁|ψ₂⟩|²
        """
        # Esencialmente el producto escalar de los vectores de estado y luego el cuadrado del valor absoluto
        producto_escalar = np.vdot(estado1.vector, estado2.vector)
        return abs(producto_escalar) ** 2

    def evaluar_calidad(self, fidelidad: float) -> str:
        """Evalúa la calidad de la reconstrucción basada en la fidelidad promedio."""
        if fidelidad > 0.95:
            return "EXCELENTE"
        elif fidelidad > 0.85:
            return "BUENA"
        elif fidelidad > 0.70:
            return "ACEPTABLE"
        else:
            return "POBRE"

    def actualizar_metricas_sistema(self):
        """Actualiza las métricas del sistema para el siguiente ciclo, simulando cambios en el entorno."""
        self.metricas.ciclo += 1
        
        # Simular fluctuaciones térmicas (afectan directamente la decoherencia)
        self.metricas.temperatura_mk += np.random.normal(0, 0.5) # Variación de 0.5 mK
        self.metricas.temperatura_mk = max(10.0, self.metricas.temperatura_mk) # Temperatura mínima de 10 mK

        # Actualizar tiempo de decoherencia T2 y T1 basado en temperatura
        # Modelado simple: mayor temperatura, menor tiempo de coherencia
        self.metricas.decoherencia_t2_us = max(10.0, 200.0 / (self.metricas.temperatura_mk / 15.0)) # T2 inversamente proporcional a T
        self.metricas.decoherencia_t1_us = max(20.0, 400.0 / (self.metricas.temperatura_mk / 15.0)) # T1 también afectado

        # Simular variaciones en el ruido del canal
        self.metricas.potencia_ruido_dbm += np.random.normal(0, 1.0) # Variación de 1 dBm
        self.metricas.potencia_ruido_dbm = min(-60.0, self.metricas.potencia_ruido_dbm) # Límite superior de ruido

        # Simular fluctuaciones en QBER basado en ruido y decoherencia
        noise_impact = (self.metricas.potencia_ruido_dbm + 90.0) / 30.0 # Normalizado (0 a 1)
        decoherence_impact = 1 - (self.metricas.decoherencia_t2_us / 200.0) # Normalizado (0 a 1)
        self.metricas.qber_estimado = max(0.001, min(0.5, (noise_impact + decoherence_impact) / 2 + np.random.normal(0, 0.005)))
