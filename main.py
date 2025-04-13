import json
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
import itertools
import math
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

class SistemaDeTransporteIA:
    def __init__(self, datos_archivo: str):
        with open(datos_archivo, 'r', encoding='utf-8') as archivo:
            datos = json.load(archivo)

        self.estaciones = {est['id']: est for est in datos['estaciones']}
        self.conexiones = datos.get('conexiones', [])
        self.reglas = datos.get('reglas', [])

        self.grafo = self._construir_grafo_networkx()
        self.clusterizador = self._crear_modelo_clustering()

        self.q_learning = QLearningRouter(self.grafo)

    def _construir_grafo_networkx(self) -> nx.DiGraph:
        G = nx.DiGraph()

        for conexion in self.conexiones:
            if not self._validar_conexion(conexion):
                continue

            origen = conexion['origen']
            destino = conexion['destino']
            peso = self._calcular_peso_conexion(conexion)

            G.add_edge(origen, destino,
                      tiempo=conexion.get('tiempo', 10),
                      distancia=conexion.get('distancia', 1),
                      linea=conexion.get('linea', 'desconocida'),
                      peso=peso)

        return G

    def _validar_conexion(self, conexion: Dict) -> bool:
        for regla in self.reglas:
            if regla['tipo'] == 'mantenimiento_tramo':
                if regla['origen'] == conexion['origen'] and regla['destino'] == conexion['destino']:
                    return False
        return True

    def _calcular_peso_conexion(self, conexion: Dict) -> float:
        tiempo = conexion.get('tiempo', 10)
        distancia = conexion.get('distancia', 1)

        for regla in self.reglas:
            if regla['tipo'] == 'congestion' and regla['linea'] == conexion.get('linea'):
                tiempo *= regla.get('factor', 1)

        return tiempo / distancia

    def _crear_modelo_clustering(self) -> KMeans:
        X_train = self._preparar_datos_entrenamiento()
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_train)

        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X_scaled)

        self.cluster_labels = kmeans.labels_
        return kmeans

    def _preparar_datos_entrenamiento(self) -> np.ndarray:
        caracteristicas = []

        for conexion in self.conexiones:
            caracteristica = [
                conexion.get('distancia', 1),
                conexion.get('tiempo', 10),
                len(self.estaciones[conexion['origen']]['lineas'])
            ]
            caracteristicas.append(caracteristica)

        return np.array(caracteristicas)

    def encontrar_ruta_ia(self, origen: str, destino: str) -> Dict:
        ruta_q_learning = self.q_learning.encontrar_ruta(origen, destino)

        tiempos_estimados = []
        for i in range(len(ruta_q_learning) - 1):
            origen_ruta = ruta_q_learning[i]
            destino_ruta = ruta_q_learning[i+1]

            tiempo_base = self.grafo[origen_ruta][destino_ruta]['tiempo']

            caracteristicas = np.array([
                self.grafo[origen_ruta][destino_ruta]['distancia'],
                tiempo_base,
                len(self.estaciones[origen_ruta]['lineas'])
            ]).reshape(1, -1)

            scaler = MinMaxScaler()
            caracteristicas_scaled = scaler.fit_transform(caracteristicas)

            grupo = self.clusterizador.predict(caracteristicas_scaled)[0]

            # Ajustar tiempo según el grupo
            factor_grupo = {0: 1.0, 1: 1.2, 2: 0.8}
            tiempo_estimado = max(5, tiempo_base * factor_grupo.get(grupo, 1))
            tiempos_estimados.append(tiempo_estimado)

        tiempo_total = sum(tiempos_estimados)
        horas_total = int(tiempo_total // 60)
        minutos_total = int(tiempo_total % 60)

        tiempos_tramos_formato = []
        for tiempo in tiempos_estimados:
            horas = int(tiempo // 60)
            minutos = int(tiempo % 60)
            tiempos_tramos_formato.append({"horas": horas, "minutos": minutos})

        return {
            "origen": origen,
            "destino": destino,
            "ruta": ruta_q_learning,
            "tiempo_total": {"horas": horas_total, "minutos": minutos_total},
            "tiempos_tramos": tiempos_tramos_formato,
            "detalles_estaciones": [self.estaciones[est] for est in ruta_q_learning]
        }

class QLearningRouter:
    def __init__(self, grafo: nx.DiGraph, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.grafo = grafo
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self._inicializar_q_table()

    def _inicializar_q_table(self):
        for nodo in self.grafo.nodes():
            for vecino in self.grafo.neighbors(nodo):
                if (nodo, vecino) not in self.q_table:
                    self.q_table[(nodo, vecino)] = np.random.uniform(0, 1)

    def encontrar_ruta(self, origen: str, destino: str) -> List[str]:
        ruta_actual = [origen]
        nodo_actual = origen
        max_iteraciones = len(self.grafo.nodes) * 2
        iteraciones = 0

        while nodo_actual != destino and iteraciones < max_iteraciones:
            vecinos = list(self.grafo.neighbors(nodo_actual))
            if not vecinos:
                break

            if np.random.random() < self.epsilon:
                siguiente_nodo = np.random.choice(vecinos)
            else:
                valores_q = [self.q_table.get((nodo_actual, vecino), 0) for vecino in vecinos]
                siguiente_nodo = vecinos[np.argmax(valores_q)]

            ruta_actual.append(siguiente_nodo)
            nodo_actual = siguiente_nodo
            iteraciones += 1

        return ruta_actual

def convertir_numpy_a_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convertir_numpy_a_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convertir_numpy_a_json(v) for v in obj]
    return obj

def main():
    inicio = time.time()

    print("🚉 Iniciando Sistema de Transporte con IA (No Supervisado)...")

    sistema = SistemaDeTransporteIA('datos.json')

    ids_estaciones = list(sistema.estaciones.keys())
    rutas_ejemplo = list(itertools.permutations(ids_estaciones, 2))

    print(f"Total de rutas a calcular: {len(rutas_ejemplo)}")

    resultados = []
    for i, (origen, destino) in enumerate(rutas_ejemplo, 1):
        if origen != destino:
            print(f"Calculando ruta {i}/{len(rutas_ejemplo)}: {origen} → {destino}")
            ruta = sistema.encontrar_ruta_ia(origen, destino)
            resultados.append(ruta)

    resultados_json = convertir_numpy_a_json(resultados)

    with open('Resultados.json', 'w', encoding='utf-8') as f:
        json.dump(resultados_json, f, ensure_ascii=False, indent=2)

    fin = time.time()
    tiempo_ejecucion = fin - inicio

    print(f"\n✅ Proceso completado.")
    print(f"🕒 Tiempo total de ejecución: {tiempo_ejecucion:.2f} segundos")
    print(f"📄 Resultados guardados en 'Resultados.json'")

if __name__ == "__main__":
    main()
