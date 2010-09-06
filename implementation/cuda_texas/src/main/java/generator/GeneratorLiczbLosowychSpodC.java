package generator;

import cuda.swig.SWIGTYPE_p_GeneratorLosowych;
import cuda.swig.generator_liczb_losowych;
import engine.TexasSettings;

public class GeneratorLiczbLosowychSpodC {

    SWIGTYPE_p_GeneratorLosowych generator;
    
    public GeneratorLiczbLosowychSpodC() {
        TexasSettings.setTexasLibraryPath();
        generator = generator_liczb_losowych.getGeneratorLosowych();
    }
    
    public int nextInt() {
        return generator_liczb_losowych.nextInt( generator );
    }
    
}
