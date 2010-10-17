#include "../struktury/generatorLosowych.h"



extern "C" {


	extern void destruktorGeneratoraLosowych(GeneratorLosowych *generator) {
		free(generator);
	};
	extern GeneratorLosowych *getGeneratorLosowych() {
		GeneratorLosowych *generator = new GeneratorLosowych();
		generator->m_z=7542;
		generator->m_w=92465;
		return generator;
	};
	extern GeneratorLosowych *getGeneratorLosowychWithSeed(int initNumber) {
		GeneratorLosowych *generator = new GeneratorLosowych();
		generator->m_z=7542*initNumber+7;
		generator->m_w=92465*initNumber+3;
		return generator;
	};

}


