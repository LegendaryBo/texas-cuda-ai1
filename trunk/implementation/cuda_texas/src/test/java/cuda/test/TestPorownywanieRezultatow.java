package cuda.test;

import engine.RegulyGry;
import engine.TexasSettings;
import engine.rezultaty.Rezultat;
import generator.GeneratorRozdan;
import junit.framework.TestCase;
import cuda.swig.SWIGTYPE_p_Rozdanie;
import cuda.swig.ai_texas_swig;
import cuda.swig.texas_swig;

/**
 * 
 * Sprawdzamy, czy
 * 
 * @author Kacper Gorski
 */
public class TestPorownywanieRezultatow extends TestCase {

	final int ILOSC_SPRAWDZEN = 20000;

	public void setUp() {
		TexasSettings.setTexasLibraryPath();	
	}
	
	public static int ile_porownan=0;
	
	/**
	 * Test losuje pewna ilosc rozdan i sprawdza, czy program w C i javie obliczyly tak samo
	 */
	public void testPorownywaniarezultatow() {
		
		System.out.println("\n\n== Test porownywania handow ("+ILOSC_SPRAWDZEN*36+" testow) ==");
		System.out.print("sprawdzenie ");
		
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		for (int i=0; i < ILOSC_SPRAWDZEN; i++) {
			int[] seed = new int[1];
			seed[0] = gneratorRozdan.getSeed();
			
			SWIGTYPE_p_Rozdanie rozdanie = texas_swig.gerRozdaniePtr();
			texas_swig.generuj(seed[0], rozdanie, seed);
			
			texas_swig.sprawdzRezultatyHOST(rozdanie);

			for (int k=0; k < 6; k++) {
				for (int j=0; j < 6; j++) {
					porownaj(gneratorRozdan, rozdanie, k, j);
				}	
			}
			

			ai_texas_swig.destruktorRozdanie(rozdanie);	
			gneratorRozdan.generate();
			
	

			
		}		
			
		System.out.print(" Test zakonczony sukcesem");
	}



	private void porownaj(GeneratorRozdan gneratorRozdan,
			SWIGTYPE_p_Rozdanie rozdanie, int i, int j) {
		
		
		Rezultat rezultat1 =  RegulyGry.najlepsza_karta( gneratorRozdan.getAllCards(i) );
		Rezultat rezultat2 =  RegulyGry.najlepsza_karta( gneratorRozdan.getAllCards(j)  );
		
		ile_porownan++;
		if (ile_porownan%120000==0)
			System.out.print("... "+ile_porownan);
		
//		System.out.println("sprawdzenie "+i);
//		System.out.println(rezultat1);
//		System.out.println(rezultat2);
		
		assertEquals(rezultat1.porownaj(rezultat2) , texas_swig.porownaj(rozdanie, i, j));
	}

	
}
