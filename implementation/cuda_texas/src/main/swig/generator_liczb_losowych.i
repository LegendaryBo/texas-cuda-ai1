%module generator_liczb_losowych
%{
#include "../../classes/struktury/generatorLosowych.h"


extern void destruktorGeneratoraLosowych(GeneratorLosowych *generator);
extern GeneratorLosowych *getGeneratorLosowych();
extern int nextInt(GeneratorLosowych *generator);



%}

%include typemaps.i 

extern void destruktorGeneratoraLosowych(GeneratorLosowych *IN);
extern GeneratorLosowych *getGeneratorLosowych();
extern int nextInt(GeneratorLosowych *IN);

