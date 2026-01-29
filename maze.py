import random
import os

# ==============================
# GENERADOR DE LABERINTO RANDOM
# ==============================
def generar_laberinto(filas=11, columnas=18):
    lab = [[1 for _ in range(columnas)] for _ in range(filas)]

    def vecinos(x, y):
        dirs = [(-2,0),(2,0),(0,-2),(0,2)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 1 <= nx < filas-1 and 1 <= ny < columnas-1:
                yield nx, ny, dx, dy

    stack = [(1, 1)]
    lab[1][1] = 0

    while stack:
        x, y = stack[-1]
        for nx, ny, dx, dy in vecinos(x, y):
            if lab[nx][ny] == 1:
                lab[x + dx//2][y + dy//2] = 0
                lab[nx][ny] = 0
                stack.append((nx, ny))
                break
        else:
            stack.pop()

    lab[1][1] = "S"
    lab[filas-2][columnas-2] = "E"
    return lab


# ==============================
# CLASE LABERINTO (POO)
# ==============================
class Laberinto:
    def __init__(self, matriz):
        self.matriz = matriz
        self.posicion = self.encontrar_inicio()

    def encontrar_inicio(self):
        for i in range(len(self.matriz)):
            for j in range(len(self.matriz[0])):
                if self.matriz[i][j] == "S":
                    return (i, j)

    def mover(self, direccion):
        x, y = self.posicion

        if direccion == "w":
            nuevo = (x - 1, y)
        elif direccion == "s":
            nuevo = (x + 1, y)
        elif direccion == "a":
            nuevo = (x, y - 1)
        elif direccion == "d":
            nuevo = (x, y + 1)
        else:
            return

        if self.es_valido(nuevo):
            self.posicion = nuevo

    def es_valido(self, pos):
        x, y = pos
        return (
            0 <= x < len(self.matriz)
            and 0 <= y < len(self.matriz[0])
            and self.matriz[x][y] != 1
        )

    def llego_salida(self):
        x, y = self.posicion
        return self.matriz[x][y] == "E"


# ==============================
# MOSTRAR LABERINTO
# ==============================
def mostrar_laberinto(lab):
    os.system("cls" if os.name == "nt" else "clear")
    for i in range(len(lab.matriz)):
        fila = ""
        for j in range(len(lab.matriz[0])):
            if (i, j) == lab.posicion:
                fila += "🧍 "
            elif lab.matriz[i][j] == 1:
                fila += "⬛ "
            elif lab.matriz[i][j] == 0:
                fila += "⬜ "
            elif lab.matriz[i][j] == "S":
                fila += "🚪 "
            elif lab.matriz[i][j] == "E":
                fila += "🏁 "
        print(fila)
    print("\nW↑ S↓ A← D→ | Q salir")


# ==============================
# JUEGO
# ==============================
maze = generar_laberinto()
lab = Laberinto(maze)

print("🎮 LABERINTO RANDOM 11x18")
input("Presiona ENTER para empezar")

while True:
    mostrar_laberinto(lab)

    if lab.llego_salida():
        print("🎉 ¡FELICIDADES! Llegaste a la salida")
        break

    movimiento = input("Movimiento: ").lower()

    if movimiento == "q":
        print("Juego terminado 👋")
        break

    lab.mover(movimiento)