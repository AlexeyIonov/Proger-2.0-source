#include<iostream>
#include<immintrin.h>
#include<fstream>
#include<time.h>
#include<chrono>
using namespace std;


class nn {
public:
	int layersN;
	int* arch;
	string *activationFunctions;
	long long neuronsNum = 0;
	long long weightsNum = 0;
	double* values;	//values of neurons
	double* errors;
	double* weights; // weights of neurons
	nn(int lN, int *Arch, string *AF) {
		layersN = lN;
		arch = new int[lN];
		activationFunctions = new string[lN];
		for (int i = 0; i < lN; i++) {
			arch[i] = Arch[i];
			activationFunctions[i] = AF[i];
			neuronsNum += arch[i];
			if (i > 0) {
				weightsNum += arch[i] * arch[i - 1];
			}
		}
		weights = new double[weightsNum];
		values = new double[neuronsNum];
		errors = new double[neuronsNum];
	}

	void gemm(int M, int N, int K, const double* A, const double* B, double* C)
	{
		for (int i = 0; i < M; ++i)
		{
			double* c = C + i * N;
			for (int j = 0; j < N; ++j)
				c[j] = 0;
			for (int k = 0; k < K; ++k)
			{
				const double* b = B + k * N;
				float a = A[i * K + k];
				for (int j = 0; j < N; ++j)
					c[j] += a * b[j];
			}
		}
	}
	void gemmSum(int M, int N, int K, const double* A, const double* B, double* C)
	{
		for (int i = 0; i < M; ++i)
		{
			double* c = C + i * N;
			for (int k = 0; k < K; ++k)
			{
				const double* b = B + k * N;
				float a = A[i * K + k];
				for (int j = 0; j < N; ++j)
					c[j] += a * b[j];
			}
		}
	}

	void New() {
		long long counter = 0;
		for (int i = 0; i < layersN - 1; i++) {
			for (int j = 0; j < arch[i] * arch[i + 1]; j++) {
				weights[counter] = (double(rand() % 101) / 100.0) / arch[i + 1];
				counter++;
			}
		}
	}

	void act(int valuesC, int layer) {
		if (activationFunctions[layer] == "sigmoid") {
			for (int i = valuesC; i < valuesC + arch[layer]; i++) {
				values[i] = (1 / (1 + pow(2.71828, -values[i])));
			}

		}
		if (activationFunctions[layer] == "relu") {
			for (int i = valuesC; i < valuesC + arch[layer]; i++) {
				if (values[i] < 0) values[i] = 0;
				else values[i] *= 0.01;
			}
		}
		if (activationFunctions[layer] == "softmax") {
			double zn = 0.0;
			for (int i = 0; i < arch[layer]; i++) {
				zn += pow((2.71), values[valuesC + i]);
			}
			for (int i = 0; i < arch[layer]; i++) {
				values[valuesC + i] = pow(2.71, values[valuesC + i]) / zn;
			}
		}
	}
	
	void pro(double* value, int ecounter, int layer) {
		if (activationFunctions[layer] == "sigmoid") {
			for (int i = 0; i < arch[layer]; i++) {
				values[ecounter + i] = values[ecounter + i] * (1.0 - values[ecounter + i]);
				value[i] *= values[ecounter + i];
			}
		}

		if (activationFunctions[layer] == "relu") {
			for (int i = 0; i < arch[layer]; i++) {
				if (values[i + ecounter] < 0) values[i] = 0.0;
				else values[i + ecounter] = 0.01;
				value[i] *= values[ecounter + i];
			}
		}

		if (activationFunctions[layer] == "softmax") {
			/*
			float** matrix = new float *[arch[layer]];
			for (int i = 0; i < arch[layer]; i++) {
				matrix[i] = new float[arch[layer]];
			}
			for (int i = 0; i < arch[layer]; i++) {
				for (int j = 0; j < arch[layer]; j++) {
					if (i == j) {
						matrix[i][j] = values[ecounter + i] * (1.0 - values[ecounter + i]);
					}
					else {
						matrix[i][j] = -(values[ecounter + i] * values[ecounter + j]);
					}
				}
			}
			float* pros = new float[arch[layer]];
			for (int i = 0; i < arch[layer]; i++) {
				pros[i] = 0.0;
				for (int j = 0; j < arch[layer]; j++) {
					pros[i] += matrix[j][i];
				}
				value[i] *= pros[i];

			}
			for (int i = 0; i < arch[layer]; i++) {
				delete[] matrix[i];
			}
			delete[] pros;*/
			for (int i = 0; i < arch[layer]; i++) {
				value[i] *= values[ecounter + i] * (1.0 - values[ecounter + i]);
			}
		}
	}

	void ForwardFeed(double* input_data) {
		for (int i = 0; i < arch[0]; i++) {
			values[i] = input_data[i];
		}

		long long valuesC = 0;
		long long weightsC = 0;
		for (int i = 0; i < layersN - 1; i++) {
			double* a = values + valuesC;
			double* b = weights + weightsC;
			double* c = values + valuesC + arch[i];
			gemm(1, arch[i + 1], arch[i], a, b, c);


			//for (int j = valuesC; j < valuesC + arch[i + 1]; j++) {
			valuesC += arch[i];
			act(valuesC, i + 1);

			

			weightsC += arch[i] * arch[i + 1];
			
		}

	}

	void GetPrediction(double *result) {
		long long h = neuronsNum - arch[layersN - 1];
		float sum = 0;
		for (int i = 0; i < arch[layersN - 1]; i++) {
			result[i] = values[h + i];
			sum += result[i];
			//cout << "Result[" << i << "]: " << result[i] << endl;
		}
	}

	void BackPropogation(double* rightResults, float lr) {
		//Сначала вычисление ошибок
		int h = neuronsNum - arch[layersN - 1];
		for (int i = 0; i < arch[layersN - 1]; i++) {
			errors[i + h] = rightResults[i] - values[i + h];
		}
		long long wcounter = weightsNum;
		long long ecounter = neuronsNum;
		long long counter = neuronsNum - arch[layersN - 1];
		for (int i = layersN - 1; i > 0; i--) {
			ecounter -= arch[i];
			wcounter -= arch[i] * arch[i - 1];
			counter -= arch[i - 1];
			double* a = errors + ecounter;
			double* b = weights + wcounter;
			double* c = errors + counter;
			gemm(1, arch[i - 1], arch[i], a, b, c);

		}

		//Потом обновление весов:
		long long vcounter = neuronsNum - arch[layersN - 1];
		wcounter = weightsNum;
		ecounter = neuronsNum;
		for (int i = layersN - 1; i > 0; i--) {
			ecounter -= arch[i];
			vcounter -= arch[i - 1];
			wcounter -= arch[i] * arch[i - 1];
			double* b = new double[arch[i]];
			for (int j = 0; j < arch[i]; j++) {
				b[j] = errors[ecounter + j] /*pro(values[ecounter + j], i)*/ * lr;
			}
			pro(b, ecounter, i);
			double* a = values + vcounter;
			double* c = weights + wcounter;

			gemmSum(arch[i - 1], arch[i], 1, a, b, c);
			
			delete[] b;
		}

	}

	void SaveWeights(string filename) {
		ofstream fout;
		fout.close();
		fout.open(filename);
		for (int i = 0; i < weightsNum; i++) {
			fout << weights[i] * 10000.0 << " ";
		}
		fout.close();
	}
};

struct data_one {
	double info[28*28];
	int rresult;
};

int main() {
	
	srand(time(0));
	setlocale(LC_ALL, "Russian");
	const int N = 3;
	int size[N] = { 28*28, 784, 10 };
	string act[N] = { "-" , "relu", "softmax" };
	nn a(N, size, act);
	a.New();
	
	ifstream fin("mnist/bin/x_train_bin.bin", ios::binary);

	const int n = 60000;
	data_one *data = new data_one[n];

	double ra = 0.0;

	const int epochN = 15;

	double input[28*28];
	int rresult;
	
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < 28*28; j++) {
			unsigned char c;
			fin.read((char*)&c, 1);
			data[i].info[j] = double(int(c)) / 255.0;
		}
		//data[i].rresult -= 65;
	}
	fin.close();
	fin.open("mnist/y_train.txt");
	for (int i = 0; i < n; i++) {
		fin >> data[i].rresult;
	}
	cout << "Считывание завершено. Начало обучения!\n";
	double* result = new double[size[N - 1]];
	double* rresults = new double[size[N - 1]];
	auto start = clock();
	for (int e = 0; e < epochN; e++) {
		auto epoch_start = chrono::high_resolution_clock::now();
		cout << "Эпоха # " << e + 1 << endl;
		ra = 0;
		for (int i = 0; i < n; i++) {
			/*
			for (int j = 0; j < 4096; j++) {
				input[j] = data[i].info[j];
			}*/
			rresult = data[i].rresult;

			//cout << int(rresult) << endl;
			//cout << "Цифра " << rresult << endl;

			a.ForwardFeed(data[i].info);

			//cout << "ForwardFeed Time: " << FF_stop - FF_start << endl;
			//nn.show();


			a.GetPrediction(result);
			double predictionv = 0;
			int predictionIndex = 0;
			for (int m = 0; m < size[N-1]; m++) {
				if (predictionv < result[m]) {
					predictionv = result[m];
					predictionIndex = m;
				}
			}

			

			//cout << "Предсказание нейросети: " << char(predictionIndex + 65) << endl;

			if (predictionIndex == rresult) {
				//cout << "Результат верный!\n";
				//cout << "Угадал букву " << char(rresult + 65) << endl;
				ra++;
			}
			else {
				//cout << "Результат " << result << " неверный!\n";
				//cout << "Не угадал букву " << char(rresult + 65) << "\n";

				for (int m = 0; m < size[N-1]; m++) {
					rresults[m] = 0.0;
				}
				rresults[data[i].rresult] = 1.0;
				a.BackPropogation(rresults, 0.05);

				//cout << "BackPropogation time: " << BP_stop - BP_start << endl;
			}
		}
		auto epoch_stop = chrono::high_resolution_clock::now();
		cout << "Время: " << chrono::duration_cast<std::chrono::seconds>(epoch_stop - epoch_start).count() << " seconds\t";
		cout << "Правильных ответов: " << (ra / double(n)) * 100 << "%" << endl;
		if ((e > 0) and (((ra / double(n)) * 100) < 10)) {
			for (int m = 0; m < size[N - 1]; m++) {
				cout << result[m] << endl;
			}
			cout << "Errors:\n";
			/*for (int m = 0; m < size[N - 2]; m++) {
				cout << "a.errors[" << a.neuronsNum - size[N - 1] - size[N - 2] + m << "]: " << a.errors[a.neuronsNum - size[N - 1] - size[N - 2] + m] << endl;
			}*/
		}
	}
	
	cout << "Time: " << clock() - start << endl;
	a.SaveWeights("weights.txt");

	/*
	int w = 0;
	int v = 0;
	for (int i = 0; i < 4; i++) {
		cout << "Layer #" << i << endl;
		for (int j = 0; j < a.arch[i]; j++) {
			cout << "Neuron #" << j << ". Value: " << a.values[v] << endl;
			for (int k = 0; k < a.arch[i + 1]; k++) {
				cout << "Weight # " << k << " : " << a.weights[w] << endl;
				w += 1;

			}
			v += 1;
		}
	}*/

	cout << "Начинаю тест:\n";
	data_one *test_data = new data_one[10000];
	fin.close();
	fin.open("mnist/x_test.txt");
	for (int test = 0; test < 10000; test++) {
		for (int i = 0; i < 28 * 28; i++) {
			fin >> test_data[test].info[i];
			test_data[test].info[i] /= 255.0;
		}
	}
	fin.close();
	fin.open("mnist/y_test.txt");
	for (int test = 0; test < 10000; test++) {
		fin >> test_data[test].rresult;
	}

	int rightAnswersCounter = 0;
	auto testStart = chrono::high_resolution_clock::now();
	for (int test = 0; test < 10000; test++) {
		
		a.ForwardFeed(test_data[test].info);
		double test_result[10];
		a.GetPrediction(test_result);

		
		double predictionv = 0;
		int predictionIndex = 0;
		for (int m = 0; m < size[N - 1]; m++) {
			if (predictionv < test_result[m]) {
				predictionv = test_result[m];
				predictionIndex = m;
			}
		}
		if (predictionIndex == test_data[test].rresult) {
			//cout << "Угадал букву " << char(predictionIndex + 65) << endl;
			rightAnswersCounter++;
		}
	}
	auto testStop = chrono::high_resolution_clock::now();
	cout << "Time: " << chrono::duration_cast<std::chrono::seconds>(testStop - testStart).count() << " ms\n";
	cout << "Правильных ответов: " << double(rightAnswersCounter) / 10000.0 * 100 << "%"<< endl;
	
	return 0;
}