# bimotype_v2/main.py

from core.datatypes import MetricasSistema
from components.classical import AIEngine
from components.quantum import Qubit, Transductor
from protocols.bimotype import BiMoTypeV2Protocol

class Simulador:
    def __init__(self):
        print("Inicializando Simulador BiMoType v2.0...")
        self.metricas = MetricasSistema(ciclo=0, temperatura_mk=15.0, ...) # Inicializar métricas
        self.ia = AIEngine()
        self.protocolo = BiMoTypeV2Protocol()
        self.qubit_transmisor = Qubit("Q-Tx", self.metricas)
        self.transductor = Transductor(self.metricas)

    def ciclo_completo(self, comando: str):
        print(f"\n--- INICIANDO CICLO DE SIMULACIÓN PARA: '{comando}' ---")
        
        # 1. Capa de IA: Interpretar comando
        carga_util = self.ia.analizar_comando(comando)
        
        # 2. Capa de Protocolo: Codificar paquete completo
        paquete = self.protocolo.encode("msg-001", carga_util)
        
        # 3. Capa de Modulación: Convertir paquete a pulsos (conceptual)
        secuencia_pulsos = self.protocolo.to_pulse_sequence(paquete)
        
        # --- Aquí simularíamos la transmisión y recepción ---
        print("Simulando transmisión por canal óptico y radiactivo...")
        
        # 4. Capa de Demodulación: Reconstruir carga útil a partir de pulsos
        carga_recibida = self.protocolo.from_pulse_sequence(secuencia_pulsos)

        # 5. Verificación
        if carga_recibida:
            print("✅ ¡Éxito! El ciclo de comunicación se completó y los datos se reconstruyeron.")
        else:
            print("❌ ¡Fallo! Hubo un error en la decodificación.")

if __name__ == "__main__":
    sim = Simulador()
    sim.ciclo_completo("simulate entanglement_bell_pair --shots 1024")
