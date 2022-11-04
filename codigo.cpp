////////////////////////////////Cabeceras/////////////////////////////////////
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
/////////////////////////////////////////////////////////////////////////////

///////////////////////////////Espacio de nombres////////////////////////////
using namespace cv;
using namespace std;
#define PI 3.14159265358979323846
/////////////////////////////////////////////////////////////////////////////

float formulaGaussiana(int x, int y, int sigma) {
	float resultado;
	float parte1 = 1 / (2 * PI * pow(sigma, 2));
	float parte2 = (-(pow(x, 2) + pow(y, 2))) / (2 * pow(sigma, 2));

	resultado = parte1 * exp(parte2);
	return resultado;
}

float** crearKernel(int tamanho, int restante, int sigma) {
	//Creación matriz
	float** matriz = new float* [tamanho];

	for (int i = 0; i < tamanho; i++) {
		matriz[i] = new float[tamanho];
	}

	//Indica posiciones y crea kernel
	int fila;
	int columna = restante; 

	for (int i = 0; i < tamanho; i++) {
		fila = -restante; //Auxiliares de posiciones del kernel, donde el centro es (0,0)
		for (int j = 0; j < tamanho; j++) {
			matriz[i][j] = formulaGaussiana(fila, columna, sigma);
			cout << matriz[i][j] << "\t";
			fila = fila + 1;
		}
		columna = columna - 1;
		cout << endl;
	}
	return matriz;
}

Mat imagenGrises() {
	//Obtener y mostrar imagen original (RGB)
	char NombreImagen[] = "C:\\ImagenesVA\\lena.jpg";
	Mat imagenOriginal;

	imagenOriginal = imread(NombreImagen);

	if (!imagenOriginal.data)
	{
		cout << "Error al cargar la imagen: " << NombreImagen << endl;
		exit(1);
	}

	//Creamos una ventana para mostrar la imagen
	namedWindow("Imagen original", WINDOW_AUTOSIZE);
	imshow("Imagen original", imagenOriginal);

	//Conversión por método promedio
	double azul, verde, rojo;
	double promedio, valorTruncado;

	int fila_original = imagenOriginal.rows;
	int columna_original = imagenOriginal.cols;

	cout << "Imagen Original:" << "\nFilas: " << fila_original << "\nColumnas: " << columna_original << endl;

	Mat imagen(fila_original, columna_original, CV_8UC1);

	for (int i = 0; i < fila_original; i++) {
		for (int j = 0; j < columna_original; j++) {
			//Recordar que es BGR
			azul = imagenOriginal.at<Vec3b>(Point(j, i)).val[0];
			verde = imagenOriginal.at<Vec3b>(Point(j, i)).val[1];
			rojo = imagenOriginal.at<Vec3b>(Point(j, i)).val[2];

			promedio = (azul + verde + rojo) / 3;
			valorTruncado = static_cast<int>(promedio);
			imagen.at<uchar>(Point(j, i)) = uchar(valorTruncado);
		}
	}

	fila_original = imagen.rows;
	columna_original = imagen.cols;

	cout << "Imagen en escala de grises:" << "\nFilas: " << fila_original << "\nColumnas: " << columna_original << endl;

	namedWindow("Imagen escala de grises", WINDOW_AUTOSIZE);
	imshow("Imagen escala de grises", imagen);

	return imagen;
}

Mat extenderImagen(Mat imagen, int restante) {
	//Agregamos el restante arriba, abajo, derecha e izquierda. 
	int filas = imagen.rows + (restante * 2);
	int columnas = imagen.cols + (restante * 2);

	Mat imagenExtendida(filas, columnas, CV_8UC1);

	//Los bordes tienen 0 
	for (int i = 0; i < filas; i++) {
		for (int j = 0; j < columnas; j++) {
			if (i<restante || i >= imagen.rows + restante) {
				imagenExtendida.at<uchar>(Point(j, i)) = uchar(0);
			}
			else if (j < restante || j >= imagen.cols + restante) {
				imagenExtendida.at<uchar>(Point(j, i)) = uchar(0);
			}
			else {
				imagenExtendida.at<uchar>(Point(j, i)) = imagen.at<uchar>(Point(j-restante, i-restante));
			}
		}
	}
	/* Alternativa:
	for (int i = 0; i < imagen.rows; i++) {
		for (int j = 0; j < imagen.cols; j++) {
			imagenExtendida.at<uchar>(Point(j+restante, i+restante)) = imagen.at<uchar>(Point(j, i));
		}
	}*/

	//namedWindow("Imagen extendida", WINDOW_AUTOSIZE);
	//imshow("Imagen extendida", imagenExtendida);

	return imagenExtendida;
}

Mat convolucion(Mat imagen, float** kernel, int tamanho, int restante) {

	int filas = imagen.rows;
	int columnas = imagen.cols;
	double suma;
	Mat auxiliar(filas, columnas, CV_8UC1);
	//Regresamos al tamaño original de la imagen, por ejemplo, quitamos el restante de columnas a la derecha e izquierda 
	Mat imagenSuavizada(filas - (restante * 2), columnas - (restante * 2), CV_8UC1);


	for (int i = 0; i < filas; i++) {
		if ((i < (filas - restante) && i >= restante)) {

			for (int j = 0; j < columnas; j++) {
				if ((j < (columnas - restante) && j >= restante)) {
					suma = 0;
					//Recorrido del kernel
					for (int a = 0; a < tamanho; a++) {

						for (int b = 0; b < tamanho; b++) {
							suma = suma + (kernel[a][b] * imagen.at<uchar>(Point(j - restante + b, i - restante + a)));
						}
					}
					//Valor absoluto
					suma = abs(static_cast<int>(suma));
					//Regresamos a la posición central de acuerdo al kernel
					imagenSuavizada.at<uchar>(Point(j - restante, i - restante)) = uchar(suma);
				}
			}
		}


	}

	//namedWindow("Imagen suavizada", WINDOW_AUTOSIZE);
	//imshow("Imagen suavizada", imagenSuavizada);
	return imagenSuavizada;
}

float **normalizarKernel(float **kernel, int tamanho) {
	//División de cada elemento en el kernel por la suma de todos sus elementos. 
	float suma = 0;

	for (int i = 0; i < tamanho; i++) {
		for (int j = 0; j < tamanho; j++) {
			suma = suma + kernel[i][j];
		}
	}

	for (int i = 0; i < tamanho; i++) {
		for (int j = 0; j < tamanho; j++) {
			kernel[i][j] = (kernel[i][j]) / suma;
		}
	}
	return kernel;
}

Mat ecualizar(Mat imagen) {
	//Mejora el contraste de una imagen, con el fin de estirar el rango de intensidad.

	Mat imagenEcualizada;
	equalizeHist(imagen, imagenEcualizada);


	namedWindow("Imagen ecualizada", WINDOW_AUTOSIZE);
	imshow("Imagen ecualizada", imagenEcualizada);
	cout << "Imagen ecualizada:" << "\nFilas: " << imagenEcualizada.rows << "\nColumnas: " << imagenEcualizada.cols << endl;

	return imagenEcualizada;
}

void realizarHisteresis(Mat imagen) {
	float maximo = 0;

	for (int i = 0; i < imagen.rows; i++) {
		for (int j = 0; j < imagen.cols; j++) {
			if (imagen.at<uchar>(Point(j, i)) > maximo) {
				maximo = imagen.at<uchar>(Point(j, i));
			}
		}
	}
	//Obtenemos el valor más alto y tomamos nuestros umbrales
	float umbralSuperior = maximo * 0.90;
	float umbralInferior = maximo * 0.35;

	//Si la intensidad es mayor o igual al umbral superior = 255 y si es menor o igual = 0 
	for (int i = 0; i < imagen.rows; i++) {
		for (int j = 0; j < imagen.cols; j++) {

			float valor = imagen.at<uchar>(Point(j, i));

			if (valor >= umbralSuperior) {
				imagen.at<uchar>(Point(j, i)) = 255;
			}
			else if (valor <= umbralInferior) {
				imagen.at<uchar>(Point(j, i)) = 0;
			}
			else {
				imagen.at<uchar>(Point(j, i)) = 255;
			}
		}
	}	

	cout << "Imagen detecciòn de borde Canny:" << "\nFilas: " << imagen.rows << "\nColumnas: " << imagen.cols << endl;

	namedWindow("Detecciòn de borde Canny", WINDOW_AUTOSIZE);
	imshow("Detecciòn de borde Canny", imagen);
}


void suprimirMaximos(Mat imagen, Mat ang) {

	/*El valor de este ángulo se redondea a uno de los cuatro ángulos que representan la dirección vertical,
	la horizontal y las dos diagonales (0°, 45°, 90° y 135°).*/

	for (int i = 0; i < ang.rows; i++) {
		for (int j = 0; j < ang.cols; j++) {
			if (ang.at<uchar>(Point(j, i)) < 23) {
				ang.at<uchar>(Point(j, i)) = 0;
			}
			else if (ang.at<uchar>(Point(j, i)) > 23 && ang.at<uchar>(Point(j, i)) < 68) {
				ang.at<uchar>(Point(j, i)) = 45;
			}
			else if (ang.at<uchar>(Point(j, i)) > 68 && ang.at<uchar>(Point(j, i)) < 113) {
				ang.at<uchar>(Point(j, i)) = 90;
			}
			else {
				ang.at<uchar>(Point(j, i)) = 135;
			}
		}
	}
	//Si es 0°	 (fila,columna+1)   y  (fila,columna-1)
	//Si es 45°  (fila-1,columna+1) y  (fila+1,columna-1)
	//Si es 90°  (fila-1,columna)   y  (fila+1,columna) 
	//Si es 135° (fila-1,columna-1) y  (fila+1,columna+1)

	for (int i = 1; i < ang.rows-1; i++) {
		for (int j = 1; j < ang.cols-1; j++) {
			float valor = ang.at<uchar>(Point(j, i));
			float contenido = imagen.at<uchar>(Point(j, i));

			if (valor == 0) {
				//Revisamos que sea mayor a sus vecinos de acuerdo a la direcciòn obtenida
				if (contenido > imagen.at<uchar>(Point(j + 1, i)) && contenido > imagen.at<uchar>(Point(j - 1, i))) {
					imagen.at<uchar>(Point(j, i)) = imagen.at<uchar>(Point(j, i));
				}
				else {
					imagen.at<uchar>(Point(j, i)) = 0;
				}
			}
			else if (valor == 45) {
				if (contenido > imagen.at<uchar>(Point(j + 1, i - 1)) && contenido > imagen.at<uchar>(Point(j - 1, i + 1))) {
					imagen.at<uchar>(Point(j, i)) = imagen.at<uchar>(Point(j, i));
				}
				else {
					imagen.at<uchar>(Point(j, i)) = 0;
				}
			}
			else if (valor == 90) {
				if (contenido > imagen.at<uchar>(Point(j, i - 1)) && contenido > imagen.at<uchar>(Point(j, i + 1))) {
					imagen.at<uchar>(Point(j, i)) = imagen.at<uchar>(Point(j, i));
				}
				else {
					imagen.at<uchar>(Point(j, i)) = 0;
				}
			}
			else if (valor == 135) {
				if (contenido > imagen.at<uchar>(Point(j - 1, i - 1)) && contenido > imagen.at<uchar>(Point(j + 1, i + 1))) {
					imagen.at<uchar>(Point(j, i)) = imagen.at<uchar>(Point(j, i));
				}
				else {
					imagen.at<uchar>(Point(j, i)) = 0;
				}
			}
		}
	}

	realizarHisteresis(imagen);
}

void filtroSobel(Mat imagen, int restante) {
	//Creación de nuestros kernel
	float** mx = new float* [3];
	float** my = new float* [3];

	for (int i = 0; i < 3; i++) {
		mx[i] = new float[3];
		my[i] = new float[3];
	}

	//Lo definimos de esta manera, ya que la funciòn de convoluciòn pide un kernel del tipo float**
	mx[0][0] = -1;
	mx[0][1] = 0;
	mx[0][2] = 1;

	mx[1][0] = -2;
	mx[1][1] = 0;
	mx[1][2] = 2;

	mx[2][0] = -1;
	mx[2][1] = 0;
	mx[2][2] = 1;

	my[0][0] = -1;
	my[0][1] = -2;
	my[0][2] = -1;

	my[1][0] = 0;
	my[1][1] = 0;
	my[1][2] = 0;

	my[2][0] = 1;
	my[2][1] = 2;
	my[2][2] = 1;

	/*Para utilizar nuestra función de convolución requerimos volver a extender la imagen, pero el restante
	es 1 ya que nuestra mx y my son de 3x3 siempre*/
	Mat imagenA = extenderImagen(imagen, 1); 
	Mat G_x = convolucion(imagenA, mx, 3, 1); 
	Mat G_y = convolucion(imagenA, my, 3, 1);

	//|G| 					
	float G = 0;
	Mat imagenSobel(imagen.rows, imagen.cols, CV_8UC1);
	//Angulos
	Mat orientacion(imagen.rows, imagen.cols, CV_8UC1);
	float z;

	for (int i = 0; i < imagen.rows; i++) {
		for (int j = 0; j < imagen.cols; j++) {
			float a = G_x.at<uchar>(Point(j, i));
			float b = G_y.at<uchar>(Point(j, i));

			//Obtenemos el valor absoluto de ambos valores, al realizar la operaciòn de esta manera, evitamos la raiz cuadrada.
			a = abs(static_cast<int>(a));
			b = abs(static_cast<int>(b));

			imagenSobel.at<uchar>(Point(j, i)) = a + b;
			orientacion.at<uchar>(Point(i, j)) = (atan(b/a)*180)/PI; //El valor de radianes a grados
		}
	}

	namedWindow("Imagen sobel |G|", WINDOW_AUTOSIZE);
	imshow("Imagen sobel |G|", imagenSobel);
	cout << "Imagen |G|:" << "\nFilas: " << imagenSobel.rows << "\nColumnas: " << imagenSobel.cols << endl;

	suprimirMaximos(imagenSobel, orientacion);

}


/////////////////////////Inicio de la funcion principal///////////////////
int main()
{
	int tamanho, restante;
	float sigma;
	float** kernel;
	Mat imagen;

	cout << "Tamaño del kernel : ";
	cin >> tamanho;

	cout << "Valor de sigma :  ";
	cin >> sigma;

	restante = tamanho / 2; //Lo que sobresale el kernel de la matriz/img original


	kernel = crearKernel(tamanho, restante, sigma);

	imagen = imagenGrises();

	Mat imagenExtendida = extenderImagen(imagen, restante);

	float** kernelNormalizado = normalizarKernel(kernel, tamanho);

	Mat imagenSuavizada = convolucion(imagenExtendida,kernelNormalizado,tamanho, restante);
	namedWindow("Imagen suavizada", WINDOW_AUTOSIZE);
	imshow("Imagen suavizada", imagenSuavizada);
	cout << "Imagen suavizada:" << "\nFilas: " << imagenSuavizada.rows << "\nColumnas: " << imagenSuavizada.cols << endl;


	Mat imagenEcualizada = ecualizar(imagenSuavizada);

	filtroSobel(imagenEcualizada, restante);

	waitKey(0); //Función para esperar
	return 1;
}
/////////////////////////////////////////////////////////////////////////