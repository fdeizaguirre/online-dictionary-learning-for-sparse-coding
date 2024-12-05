# Inspirado en los repositorios
# https://gitlab.fing.edu.uy/tao/datos
# https://github.com/MehdiAbbanaBennani/online-dictionary-learning-for-sparse-coding/tree/master

# %%
import os
from itertools import tee

import numpy as np
from sklearn.linear_model import Lasso, LassoLars
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio

import datos.util as util


class OnlineDictionaryLearning:
    """
    Aprendizaje en línea de diccionarios para codificación sparsa.

    Esta clase implementa un algoritmo online para aprender un diccionario
    que pueda representar de forma sparsa un conjunto de datos.
    """

    def __init__(
        self,
        data: np.array,
        log_step: int = 40,
        test_batch_size: int = 1000,
        base_dir: str = ".",
    ):
        """
        Inicializa la clase OnlineDictionaryLearning.

        Args:
            data (np.array): Conjunto de datos para el aprendizaje del diccionario.
            log_step (int, optional): Cada cuantas iteraciones se registran los resultados. Por defecto 40.
            test_batch_size (int, optional): Tamaño del batch de prueba. Por defecto 1000.
            base_dir (str, optional): Directorio base para guardar registros y salidas. Por defecto ".".
        """

        self.data_gen = self.sample(data)
        self.n_obs = len(data)
        self.dim_obs = len(data[0])
        self.m = data.shape[1]

        self.log_step = log_step
        self.test_batch_size = test_batch_size

        self.base_dir = base_dir
        self.losses = []
        self.offline_loss = []
        self.objective = []
        self.cumulative_losses = []
        self.imagenes = []
        self.test_batch = iter(())
        np.random.seed(14)  # semilla para hacer pruebas comparables
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

    def sample(self, data: np.array):
        """
        Crea un generador aleatorio de muestras de datos sobre el cual iterar.

        Args:
            data (np.array): Conjunto de datos para muestreo.

        Yields:
            np.array: Una muestra aleatoria del conjunto de datos.
        """
        while True:
            permutation = list(np.random.permutation(self.n_obs))
            for idx in permutation:
                yield data[idx]

    def initialize_logs(self):
        """
        Inicializa las listas para registrar Loss, imágenes y obtiene un generado de datos de prueba.
        """
        self.losses = []
        self.offline_loss = []
        self.imagenes = []
        self.test_batch = iter(
            [next(self.data_gen) for i in range(self.test_batch_size)]
        )

    @staticmethod
    def compute_alpha(x: np.array, dic: np.array, lam, optimizer: str = "lasso"):
        """
        Calcula los coeficientes de representación esparsa para una observación dada.

        Args:
            x (np.array): Vector de observación.
            dic (np.array): Matriz del diccionario.
            lam (float): Parámetro de regularización.
            optimizer (str, optional): Método de optimización ('lasso' o 'lars'). Por defecto "lasso".

        Returns:
            np.array: Coeficientes de representación esparsa.

        Raises:
            Exception: Si NO se especifica un optimizador inválido.
        """

        if optimizer == "lasso":
            reg = Lasso(alpha=lam)
        elif optimizer == "lars":
            reg = LassoLars(alpha=lam)
        else:
            raise Exception(
                "Optimizador incorrecto, solo se aceptan 'lasso' (dafault) o 'lars' "
            )
        reg.fit(X=dic, y=x)
        return reg.coef_

    @staticmethod
    def compute_dic(A: np.array, B: np.array, D: np.array, k: int):
        """
        Actualiza el diccionario utilizando las matrices acumuladas A y B.

        Args:
            A (np.array): Matriz de coeficientes acumulada.
            B (np.array): Matriz acumulada de productos de datos.
            D (np.array): Diccionario actual.
            k (int): Número de átomos en el diccionario.

        Returns:
            np.array: Diccionario actualizado.
        """
        tolerancia = 1.0e-7
        error = 1
        o = 0
        D_nuevo = D.copy()
        # while not converged :
        while error > tolerancia and o < 10:
            for j in range(k):
                u_j = (B[:, j] - np.matmul(D, A[:, j])) / A[j, j] + D[:, j]
                D_nuevo[:, j] = u_j / max(np.linalg.norm(u_j), 1)
            D = D_nuevo / np.linalg.norm(D_nuevo, axis=0)
            error = np.linalg.norm(D_nuevo - D, ord="fro")
            o = o + 1
        return D

    @staticmethod
    def compute_A_B(
        A_prev: np.array,
        B_prev: np.array,
        x_i_batch: np.array,
        alphas: np.array,
        beta: int = 1,
    ):
        """
        Actualiza las matrices A y B utilizadas en la optimización del diccionario.

        Args:
            A_prev (np.array): Matriz A acumulada previa.
            B_prev (np.array): Matriz B acumulada previa.
            x_i (np.array): Observación actual.
            alpha_i (np.array): Coeficientes esparsos  para la observación.
            beta (int, optional): Parámetro de ponderación. Por defecto 1.

        Returns:
            tuple: Matrices A y B actualizadas.
        """

        A_curr = beta * A_prev + sum(
            [np.outer(alpha_i, alpha_i.T) for alpha_i in alphas]
        )
        B_curr = beta * B_prev + sum(
            [np.outer(x_i, alpha_i.T) for x_i, alpha_i in zip(x_i_batch, alphas)]
        )
        A_prev = A_curr
        B_prev = B_curr
        return A_curr, B_curr

    def learn(
        self,
        it: int,
        lam: float,
        k: int,
        train_batch_size: int = 1,
        optimizer: str = "lasso",
        init_A_mod: int = 1,
        init_B_mod: int = 1,
        init_D_mod: int = 1,
    ):
        """
        Ejecuta el aprendizaje del diccionario.

        Args:
            it (int): Cantidad de iteraciones.
            lam (float): Parámetro de regularización.
            k (int): Número de átomos en el diccionario.
            train_batch_size (int, optional): Tamaño del batch de entrenamiento. Por defecto 1.
            optimizer (str, optional): Método de optimización ('lasso' o 'lars'). Por defecto "lasso".
            init_A_mod (int, optional): Método de inicialización para la matriz A. Por defecto 1.
            init_B_mod (int, optional): Método de inicialización para la matriz B. Por defecto 1.
            init_D_mod (int, optional): Método de inicialización para el diccionario D. Por defecto 1.

        Returns:
            np.array: Diccionario aprendido.
        """
        assert train_batch_size >= 1, "train_batch_size tiene que ser >= 1"
        self.initialize_logs()

        # Init A
        if init_A_mod == 1:
            A_prev = np.random.randn(k, k)
            A_prev /= np.linalg.norm(A_prev)
        elif init_A_mod == 2:
            A_prev = 0.001 * np.ones((k, k))
        else:
            A_prev = 0.001 * np.identity(k)

        # Init B
        if init_B_mod == 1:
            B_prev = 0.001 * np.random.randn(self.m, k)
        if init_B_mod == 2:
            B_prev = 0.001 * np.ones((self.m, k))
        else:
            B_prev = np.zeros((self.m, k))

        # Init D
        D_prev = self.initialize_dic(k, self.m, self.data_gen, init_D_mod)

        for it_curr in tqdm(range(it)):
            x_i_batch = [next(self.data_gen) for i in range(train_batch_size)]
            alphas = [
                self.compute_alpha(x_i, D_prev, lam, optimizer=optimizer)
                for x_i in x_i_batch
            ]
            if train_batch_size == 1:
                beta = 1
            elif train_batch_size > 1:
                if (it_curr + 1) + 1 < train_batch_size:
                    theta = (it_curr + 1) * train_batch_size
                else:
                    theta = train_batch_size**2 + (it_curr + 1) - train_batch_size
                beta = (theta + 1 - train_batch_size) / (theta + 1)
            A_curr, B_curr = self.compute_A_B(
                A_prev, B_prev, x_i_batch, alphas, beta=beta
            )
            D_curr = self.compute_dic(A=A_curr, B=B_curr, D=D_prev, k=k)
            A_prev = A_curr
            B_prev = B_curr
            D_prev = D_curr

            if it_curr % self.log_step == 0:
                self.log(
                    observation=x_i_batch[0],
                    dictionary=D_curr,
                    it=it_curr,
                    lam=lam,
                    alpha=alphas[0],
                    optimizer=optimizer,
                )

        self.compute_objective()

        # Guardar diccionario
        np.savez(
            f"{self.base_dir}{os.sep}diccionario_its-{it}_lam-{lam}_k-{k}_{optimizer}_A-{init_A_mod}_B-{init_B_mod}_D-{init_D_mod}.npz",
            array1=D_curr,
        )

        mosaic = util.mosaico(np.array(D_curr.T))
        plt.figure(figsize=(10, 10))
        plt.imshow(mosaic, cmap="gray")
        plt.title(f"Atomos \n Final")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            f"{self.base_dir}{os.sep}temp_frame_final.png"
        )  # Guarda la imagen temporalmente
        plt.show()
        plt.close()
        self.imagenes.append(
            f"{self.base_dir}{os.sep}temp_frame_final.png"
        )  # Agrega el nombre a la lista

        x = np.arange(0, len(self.losses), max(len(self.losses) // 10, 1))
        xticks_labels = x * self.log_step

        plt.figure(figsize=(10, 10))
        plt.plot(self.cumulative_losses)
        plt.title(f"Loss acumulada")
        plt.xticks(x, xticks_labels)
        plt.xlabel("Iteración")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(
            f"{self.base_dir}{os.sep}Loss_acumulada.png"
        )  # Guarda la imagen temporalmente
        plt.show()
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.plot(self.losses)
        plt.xticks(x, xticks_labels)
        plt.title(f"Loss en la iteración")
        plt.xlabel("Iteración")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(
            f"{self.base_dir}{os.sep}Loss_por_iteracion.png"
        )  # Guarda la imagen temporalmente
        plt.show()
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.plot(self.offline_loss)
        plt.xticks(x, xticks_labels)
        plt.title(f"Offline loss en la iteración")
        plt.xlabel("Iteración")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(
            f"{self.base_dir}{os.sep}OfflineLoss_por_iteracion.png"
        )  # Guarda la imagen temporalmente
        plt.show()
        plt.close()

        with imageio.get_writer(
            f"{self.base_dir}{os.sep}proceso_diccionario_its-{it}_lam-{lam}_k-{k}_{optimizer}_A-{init_A_mod}_B-{init_B_mod}_D-{init_D_mod}.gif",
            mode="I",
            duration=0.5,
        ) as writer:
            for imagen in self.imagenes:
                frame = imageio.imread(imagen)
                writer.append_data(frame)
        return D_curr.T

    def log(
        self,
        observation: np.array,
        dictionary: np.array,
        it: int,
        lam: float,
        alpha: np.array,
        optimizer: str,
    ):
        """
        Registra el estado actual, incluyendo losss e imágenes.

        Args:
            observation (np.array): Observación actual.
            dictionary (np.array): Diccionario actual.
            it (int): Iteración actual.
            lam (float): Parámetro de regularización.
            alpha (np.array): Coeficientes sparsos  de la observación.
        """
        loss = self.one_loss(observation, dictionary, alpha)
        self.losses.append(loss)
        self.cumulative_losses.append(self.cumulative_loss())
        self.offline_loss.append(self.full_dataset_loss(dictionary, lam, optimizer))
        image_path = f"{self.base_dir}{os.sep}temp_frame_{it}.png"
        mosaic = util.mosaico(np.array(dictionary.T))
        plt.figure(figsize=(10, 10))
        plt.imshow(mosaic, cmap="gray")
        plt.axis("off")
        plt.title(f"Atomos \n Iteración {it}")
        plt.tight_layout()
        plt.savefig(image_path)  # Guarda la imagen temporalmente
        plt.close()
        self.imagenes.append(image_path)  # Agrega el nombre a la lista

    def cumulative_loss(self):
        """
        Calcula la loss acumulativa para las muestras observadas.

        Args:
            dictionary (np.array): Diccionario actual.

        Returns:
            float: Valor de loss acumulada.
        """
        n_observed = len(self.losses)

        return np.mean([self.losses[i] for i in range(n_observed)])

    @staticmethod
    def one_loss(x, dictionary: np.array, alpha: np.array):
        """
        Calcula la loss de reconstrucción para una observación.

        Args:
            x (np.array): Observación.
            dictionary (np.array): Diccionario.
            alpha (np.array): Coeficientes dispersos.

        Returns:
            float: loss de reconstrucción.
        """
        return np.linalg.norm(x - np.matmul(dictionary, alpha), ord=2) ** 2

    @staticmethod
    def initialize_dic(k: int, m: int, data_gen, init_D_mod: int = 0):
        """
        Inicializa la matriz del diccionario.

        Args:
            k (int): Número de átomos en el diccionario.
            m (int): Dimensión de las observaciones.
            data_gen (generator): Generador para muestreo de observaciones.
            init_D_mod (int, optional): Método de inicialización. Por defecto 0.

        Returns:
            np.array: Matriz del diccionario inicializada.
        """
        if init_D_mod == 1:
            D = np.random.randn(m, k)
            return D / np.linalg.norm(D, axis=0)
        else:
            return np.array([next(data_gen) for _ in range(k)]).T

    def observation_loss(
        self, x_i: np.array, dictionary: np.array, lam: float, optimizer: str
    ):
        """
        Calcula la loss para una observación.

        Args:
            x_i (np.array): Observación.
            dictionary (np.array): Diccionario.
            lam (float): Parámetro de regularización.

        Returns:
            float: loss de la observación.
        """
        alpha = self.compute_alpha(x_i, dictionary, lam, optimizer)
        return np.linalg.norm(x_i - np.matmul(dictionary, alpha), ord=2) ** 2

    def full_dataset_loss(self, dictionary: np.array, lam: float, optimizer):
        """
        Calcula la loss total para el conjunto de datos de prueba.

        Args:
            dictionary (np.array): Diccionario actual.
            lam (float): Parámetro de regularización.

        Returns:
            float: Pérdida total del conjunto de datos.
        """
        self.test_batch, data_gen = tee(self.test_batch)
        return np.mean(
            [
                self.observation_loss(next(data_gen), dictionary, lam, optimizer)
                for _ in range(self.test_batch_size)
            ]
        )

    def compute_objective(self):
        """
        Calcula el valor de la función objetivo a lo largo de las iteraciones.

        La función objetivo es la loss acumulativa promedio en cada iteración.
        """
        cumulateD_loss = np.cumsum(self.losses)
        self.objective = [
            cumulateD_loss[i] / (i + 1) for i in range(len(cumulateD_loss))
        ]


# %%
if __name__ == "__main__":

    from PIL import Image
    import datos.datos as datos
    import gc

    # %%
    # Para poder importar la base de datos de LUISA se debe haber descargado la misma y
    # colocado en el directorio: 'datos/img/chars/caracteres_luisa.npz'
    # La descarga se puede realizar utilizando el script 'get_chatacteres_luisa.sh'
    # y luego moviendo el archivo npz al directorio indicado.
    # Tener en cuenta que el directorio debe ser creado.

    # La versión actual de 'datos' tiene una ruta relativa al directorio donde se almacena
    # la base de datos descargada. El siguiente código maneja esta situación.
    try:
        luisa = datos.get_char_luisa()
    except:
        # Guardar el directorio actual
        directorio_original = os.getcwd()
        ruta_base = "datos"
        try:
            # Cambiar temporalmente al directorio base
            os.chdir(ruta_base)
            luisa = datos.get_char_luisa()
        finally:
            # Restaurar el directorio original
            os.chdir(directorio_original)

    # %%

    # Ejemplo con distintas configuraciones
    # Agregar variedad en la cantidad de train_batch_size
    confs = [
        {
            "a": 1,
            "b": 0,
            "d": 1,
            "opt": "lars",
            "lamd": 0.001,
            "train_b_s": 1,
            "iteraciones": 40000,
            "log": 40,
            "dict_size": 250,
        },
        {
            "a": 1,
            "b": 0,
            "d": 0,
            "opt": "lars",
            "lamd": 0.001,
            "train_b_s": 1,
            "iteraciones": 40000,
            "log": 40,
            "dict_size": 250,
        },
        {
            "a": 1,
            "b": 0,
            "d": 1,
            "opt": "lars",
            "lamd": 0.001,
            "train_b_s": 200,
            "iteraciones": 200,
            "log": 10,
            "dict_size": 250,
        },
        {
            "a": 1,
            "b": 0,
            "d": 1,
            "opt": "lasso",
            "lamd": 0.001,
            "train_b_s": 200,
            "iteraciones": 200,
            "log": 10,
            "dict_size": 250,
        },
        {
            "a": 1,
            "b": 0,
            "d": 1,
            "opt": "lars",
            "lamd": 0.01,
            "train_b_s": 200,
            "iteraciones": 200,
            "log": 10,
            "dict_size": 250,
        },
        {
            "a": 1,
            "b": 0,
            "d": 0,
            "opt": "lars",
            "lamd": 0.01,
            "train_b_s": 200,
            "iteraciones": 200,
            "log": 10,
            "dict_size": 250,
        },
        {
            "a": 1,
            "b": 0,
            "d": 1,
            "opt": "lars",
            "lamd": 0.001,
            "train_b_s": 200,
            "iteraciones": 200,
            "log": 10,
            "dict_size": 1000,
        },
        {
            "a": 1,
            "b": 0,
            "d": 0,
            "opt": "lars",
            "lamd": 0.001,
            "train_b_s": 200,
            "iteraciones": 200,
            "log": 10,
            "dict_size": 1000,
        },
        {
            "a": 1,
            "b": 0,
            "d": 0,
            "opt": "lars",
            "lamd": 0.001,
            "train_b_s": 400,
            "iteraciones": 400,
            "log": 10,
            "dict_size": 1000,
        },
    ]

    paths = []
    for conf in confs:
        base_dir = f"dict-{conf['dict_size']}_its-{conf['iteraciones']}_a-{conf['a']}_b-{conf['b']}_d-{conf['d']}_opt-{conf['opt']}_lamda-{conf['lamd']}_tbs-{conf['train_b_s']}"
        ODL = OnlineDictionaryLearning(
            luisa,
            test_batch_size=1000,
            base_dir=base_dir,
        )
        D = ODL.learn(
            it=conf["iteraciones"],
            lam=conf["lamd"],
            k=conf["dict_size"],
            optimizer=conf["opt"],
            init_A_mod=conf["a"],
            init_B_mod=conf["b"],
            init_D_mod=conf["d"],
            train_batch_size=conf["train_b_s"],
        )
        gif_name = f"proceso_diccionario_its-{conf['iteraciones']}_lam-{conf['lamd']}_k-{conf['dict_size']}_{conf['opt']}_A-{conf['a']}_B-{conf['b']}_D-{conf['d']}.gif"
        paths.append(f"{base_dir}/{gif_name}")
        gc.collect()

    # %%
    # Cargar el primer fotograma de cada GIF
    gifs = [Image.open(gif_path) for gif_path in paths]

    filas = 2
    columnas = int(np.ceil(len(gifs) / filas))
    # Leer los GIFs y obtener el primer cuadro de cada uno

    # Obtener el tamaño de un cuadro (suponemos que todos tienen el mismo tamaño)
    gif_width, gif_height = gifs[0].size

    # Crear el tamaño de la grilla
    grilla_width = columnas * gif_width
    grilla_height = filas * gif_height

    # Crear un nuevo GIF para la grilla
    grilla_frames = []  # Lista para almacenar los cuadros combinados
    frames_count = min(
        [gif.n_frames for gif in gifs]
    )  # Usar el mínimo número de cuadros

    for frame_idx in range(frames_count):
        # Crear una nueva imagen para cada cuadro de la grilla
        grilla_frame = Image.new("RGB", (grilla_width, grilla_height))

        # Pegar cada GIF en su posición dentro de la grilla
        for idx, gif in enumerate(gifs):
            gif.seek(frame_idx)  # Cambiar al cuadro actual del GIF
            fila = idx // columnas
            columna = idx % columnas
            x = columna * gif_width
            y = fila * gif_height
            grilla_frame.paste(gif, (x, y))

        grilla_frames.append(grilla_frame)

    # Guardar el nuevo GIF con la grilla
    grilla_frames[0].save(
        "grilla.gif",
        save_all=True,
        append_images=grilla_frames[1:],
        loop=1,
        duration=gifs[0].info["duration"],  # Duración del cuadro (de un GIF original)
    )
    print("GIF guardado como 'grilla.gif'")
    gc.collect()

    # %%
    ## Ejemplos de reconstrucción de caracteres con los diccionarios.
    np.random.seed(14)  # semilla para hacer pruebas comparables

    cantidad_caracteres = 10
    cantidad_diccionarios = len(confs)
    index = np.random.randint(0, luisa.shape[0], 10)
    ODL = OnlineDictionaryLearning(luisa)
    caracteres_reconstruidos = []
    for conf in confs:
        dp = f"diccionario_its-{conf['iteraciones']}_lam-{conf['lamd']}_k-{conf['dict_size']}_{conf['opt']}_A-{conf['a']}_B-{conf['b']}_D-{conf['d']}.npz"
        base_dir = f"dict-{conf['dict_size']}_its-{conf['iteraciones']}_a-{conf['a']}_b-{conf['b']}_d-{conf['d']}_opt-{conf['opt']}_lamda-{conf['lamd']}_tbs-{conf['train_b_s']}"
        dic = np.load(f"{base_dir}{os.sep}{dp}")["array1"]
        for i in index:
            caracter = luisa[i].reshape(1, -1)
            alpha = ODL.compute_alpha(caracter.T, dic, conf["lamd"], conf["opt"])
            caracter_reconstruido = dic @ alpha
            caracteres_reconstruidos.append(caracter_reconstruido)

    # Cargar diccionario de referencia y reconstruir con éste.
    diccionario_ref = np.load("datos/datos/dict/luisa_dict.npz")["dict"]
    caracteres_reconstruidos_ref = []
    for i in index:
        caracter = luisa[i].reshape(1, -1)
        alpha = ODL.compute_alpha(caracter.T, diccionario_ref.T, 0.001)
        caracter_reconstruido = alpha @ diccionario_ref
        caracteres_reconstruidos_ref.append(caracter_reconstruido)

    caracter_size = int(np.sqrt(caracter.shape[1]))
    plt.figure()
    fig, axes = plt.subplots(
        cantidad_diccionarios + 2, cantidad_caracteres, figsize=(8, 8)
    )
    plt.axis("off")
    plt.xlabel("Caracter")
    plt.ylabel("Diccionario")
    for i in range((cantidad_diccionarios + 2)):
        for j in range(cantidad_caracteres):
            if i == 0:
                axes[i, j].imshow(
                    luisa[index[j]].reshape(caracter_size, caracter_size), cmap="gray"
                )
                axes[i, j].axis("off")
            elif i == (cantidad_diccionarios + 1):
                axes[i, j].imshow(
                    caracteres_reconstruidos_ref[j].reshape(
                        caracter_size, caracter_size
                    ),
                    cmap="gray",
                )
                axes[i, j].axis("off")
            else:
                axes[i, j].imshow(
                    caracteres_reconstruidos[(i - 1) * cantidad_caracteres + j].reshape(
                        caracter_size, caracter_size
                    ),
                    cmap="gray",
                )
                axes[i, j].axis("off")
    plt.savefig(f"reconstruccion.jpeg")
    plt.show()
