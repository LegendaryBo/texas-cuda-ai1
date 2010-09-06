

#ifndef GENERATOR_LOSOWYCH_H
#define GENERATOR_LOSOWYCH_H

typedef struct {

	int m_z;
	int m_w;

	int dlugosc;
} GeneratorLosowych;



int nextInt(GeneratorLosowych *generator) {
	generator->m_z = 36969 * (generator->m_z & 65535) + (generator->m_z >> 16);
	generator->m_w = 18000 * (generator->m_w & 65535) + (generator->m_w >> 16);
	int wynik = (generator->m_z << 16) + (generator->m_w & 65535);
	if (wynik < 0)
		return -wynik;
	else
		return wynik;

}




#endif
