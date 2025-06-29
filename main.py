# bimotype_v2/main.py

from protocols.bimotype_protocol import BiMoTypeProtocol # Importar la clase renombrada
import time
import numpy as np # Necesario para np.random.normal

if __name__ == "__main__":
    print("🚀 SISTEMA UNIFICADO BiMOtype QUANTUM OS 🚀")
    print("=" * 60)

    # Inicializar sistema
    protocolo_bimotype = BiMoTypeProtocol() # Instancia de la clase BiMoTypeProtocol
    mensaje = "HELLO QUANTUM WORLD. THIS IS A TEST." # Mensaje más largo para ver más QRS

    print(f"Mensaje a transmitir: '{mensaje}'")
    print("-" * 60)

    # Codificar mensaje
    print("\n[CODIFICACIÓN]")
    paquete = protocolo_bimotype.encode_quantum_message(mensaje)
    print(f"✅ Paquete creado: {paquete.id_mensaje}")
    print(f"📊 Estados cuánticos generados (QRS): {len(paquete.estados_cuanticos)}")
    if paquete.estados_cuanticos:
        print(f"🔬 Firma radiactiva dominante (desde QRS): {paquete.firma_radiactiva.isotopo}")
    else:
        print("🔬 No se generaron estados cuánticos radiactivos para este mensaje.")

    # Simular transmisión con ruido y decoherencia
    print("\n[SIMULACIÓN DE TRANSMISIÓN Y MEDICIÓN]")
    mediciones_simuladas = []
    # Simular una medición para cada QuantumRadiationState generado
    for estado_qrs in paquete.estados_cuanticos:
        # Simular ruido en la energía medida
        # El ruido depende de la potencia de ruido del sistema
        noise_std_dev = (protocolo_bimotype.metricas.potencia_ruido_dbm + 100) / 100 * 0.2 # Escala el ruido
        energia_ruidosa = estado_qrs.energy_level + np.random.normal(0, noise_std_dev)
        mediciones_simuladas.append({'energia_medida': energia_ruidosa})
    
    print(f"✅ Simulación de {len(mediciones_simuladas)} mediciones completada")

    # Decodificar
    print("\n[DECODIFICACIÓN]")
    resultado = protocolo_bimotype.decode_quantum_transmission(paquete, mediciones_simuladas)

    # Mostrar resultados
    print("\n" + "=" * 60)
    print("📊 RESULTADOS FINALES")
    print("=" * 60)
    print(f"Mensaje Original:    '{mensaje}'")
    print(f"Mensaje Decodificado: '{resultado.get('mensaje_decodificado', 'ERROR')}'")
    print(f"Estado de Transmisión: {resultado.get('estado', 'DESCONOCIDO')}")

    metricas_cuanticas = resultado.get('metricas_cuanticas', {})
    print(f"\n📈 Métricas Cuánticas:")
    print(f"  Fidelidad Promedio: {metricas_cuanticas.get('fidelidad_promedio', 0):.1%}")
    print(f"  QBER (Tasa de error):  {metricas_cuanticas.get('qber', 0):.1%}")
    print(f"  • Estados decodificados: {metricas_cuanticas.get('estados_decodificados', 0)}")
    print(f"  • Decoherencia detectada: {'Sí' if metricas_cuanticas.get('decoherencia_detectada', False) else 'No'}")

    # Acceder a las métricas del sistema directamente del objeto protocolo_bimotype.metricas
    metricas_sistema_finales = protocolo_bimotype.metricas
    print(f"\n⚙️  MÉTRICAS DEL SISTEMA:")
    print(f"  • Ciclo de operación:    {metricas_sistema_finales.ciclo}")
    print(f"  • Temperatura:           {metricas_sistema_finales.temperatura_mk:.2f} mK")
    print(f"  • Tiempo coherencia T₂:  {metricas_sistema_finales.decoherencia_t2_us:.2f} μs")
    print(f"  • Tiempo relajación T₁:  {metricas_sistema_finales.decoherencia_t1_us:.2f} μs")
    print(f"  • Potencia de Ruido:     {metricas_sistema_finales.potencia_ruido_dbm:.2f} dBm")


    if resultado.get('errores'):
        print(f"\n⚠️  ERRORES Y ADVERTENCIAS:")
        for error in resultado['errores']:
            print(f"  • {error}")

    print("\n" + "=" * 80)
    print("🎉 DEMOSTRACIÓN COMPLETADA")
    print("=" * 80)
