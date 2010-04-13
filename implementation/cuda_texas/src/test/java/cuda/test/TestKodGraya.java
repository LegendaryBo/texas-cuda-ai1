package cuda.test;

import engine.TexasSettings;
import generator.IndividualGenerator;

import java.util.Random;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.kodGraya.KodGraya;
import cuda.swig.SWIGTYPE_p_KodGraya;
import cuda.swig.SWIGTYPE_p_int;
import cuda.swig.ai_texas_swig;

public class TestKodGraya extends TestCase {

	final int ILOSC_SPRAWDZEN = 5000;

	public void setUp() {
		TexasSettings.setTexasLibraryPath();	
	}
	
	
	/**
	 * Test losuje pewna ilosc rozdan i sprawdza, czy program w C i javie obliczyly tak samo
	 */	
	public void testKodGraya() {

		Random random = new Random(1235);
		
		System.out.println("\n\n== Test kodu graya ("+ILOSC_SPRAWDZEN+" testow) ==");
		System.out.print("sprawdzenie ");
		IndividualGenerator pIndividualGenerator = new IndividualGenerator(12, 2000);
		
		EvBinaryVectorIndividual pIndividual = pIndividualGenerator.generate();
		for (int i=0; i < ILOSC_SPRAWDZEN; i++) {
			
			for (int j=0; j < 50; j++) {

				int index_start = random.nextInt(2000);

				
				int dlugosc = random.nextInt( Math.min( 2000-index_start, 16) );
					
				int[] pOsobnikGeny = pIndividual.getGenes().clone();
		
				SWIGTYPE_p_int bla =  ai_texas_swig.getOsobnikPTR(pOsobnikGeny, pOsobnikGeny.length);
				SWIGTYPE_p_KodGraya bla2 = ai_texas_swig.getKodGrayaPTR(index_start, dlugosc);
						
				KodGraya pKod = new KodGraya(dlugosc, index_start);
						
				assertEquals(pKod.getWartoscKoduGraya(pIndividual),  ai_texas_swig.obliczKodGrayaHOST(bla, bla2) );
				
				if (i%1000==0 && j==0)
					System.out.print("... "+i);
				
				ai_texas_swig.destruktorInt(bla);
				ai_texas_swig.destruktorKodGraya(bla2);
			}	
			
		}
		System.out.print(" Test zakonczony sukcesem");
	}	
	
	
	
	
}
