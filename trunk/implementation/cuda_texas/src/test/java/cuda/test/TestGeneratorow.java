package cuda.test;

import engine.RegulyGry;
import engine.TexasSettings;
import generator.GeneratorRozdan;
import junit.framework.TestCase;
import cuda.swig.SWIGTYPE_p_Rozdanie;
import cuda.swig.ai_texas_swig;
import cuda.swig.texas_swig;

/**
 * 
 * 
 * @author Kacper Gorski (railman85@gmail.com)
 */
public class TestGeneratorow extends TestCase {

	final int ILOSC_SPRAWDZEN=50000;
	
	public void setUp() {
		TexasSettings.setTexasLibraryPath();	
	}	
	
	/**
	 * Sprawdzamy, czy generujemy takie same rozdania w C i w Javie
	 * Porownujemy wynikowe karty
	 */
	public void testGeneratoraRozdan() {

		int[] output_c = new int[6];
		int output_java;
		
		System.out.println("\n\n==Test generatora rozdan ("+ILOSC_SPRAWDZEN+" testów) ==");
		System.out.println("liczba sprawdzonych rozdan... ");
		
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan(13);
		for (int i=0; i < ILOSC_SPRAWDZEN; i++) {
			
			int[] seed = new int[1];
			seed[0] = gneratorRozdan.getSeed();

			SWIGTYPE_p_Rozdanie rozdanie = texas_swig.gerRozdaniePtr();
			texas_swig.generuj(seed[0], rozdanie, seed); // inicjalizujemy z seedem = 13
			texas_swig.najlepszaKartaRozdaniaHOST(rozdanie, output_c);	
			
			for (int j=0; j < 6; j++) {
				output_java = RegulyGry.najlepsza_karta( gneratorRozdan.getHand( j ).getKarty() ).poziom;
				assertEquals(output_java, output_c[j]);		
			}
			gneratorRozdan.generate();
			
			if (i%10000==0)
				System.out.print("... "+i+"");
			
			ai_texas_swig.destruktorRozdanie(rozdanie);
			
		}
		System.out.print(" Test zakończony sukcesem");
	}
	
}
