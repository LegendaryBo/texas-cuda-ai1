package cuda.test;

import engine.Gra;
import engine.TexasSettings;
import generator.GeneratorRozdan;
import junit.framework.TestCase;
import cuda.swig.SWIGTYPE_p_Rozdanie;
import cuda.swig.ai_texas_swig;
import cuda.swig.texas_swig;

public class TestWygrani extends TestCase {

	final int ILOSC_SPRAWDZEN = 50000;

	public void setUp() {
		TexasSettings.setTexasLibraryPath();	
	}
	
	public static int ile_porownan=0;
	
	/**
	 * Test losuje pewna ilosc rozdan i sprawdza, czy program w C i javie obliczyly tak samo
	 */
	public void testWygrani() {
	
		
		int[] wynik_c =  new int[6];
		int[] wynik_java = null;
		System.out.println("\n\n==Test sprawdzenie kart ("+ILOSC_SPRAWDZEN+" testow) ==");
		
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		for (int i=0; i < ILOSC_SPRAWDZEN; i++) {
			int[] seed = new int[1];
			seed[0] = gneratorRozdan.getSeed();
			
			SWIGTYPE_p_Rozdanie rozdanie = texas_swig.gerRozdaniePtr();
			
			texas_swig.generuj(seed[0] , rozdanie, seed );
			texas_swig.sprawdzRezultatyHOST(rozdanie);
			

			
			int ileWygranych = texas_swig.wygrany(rozdanie, new int[] {0,0,0,0,0,0}, wynik_c);

			wynik_java= Gra.sprawdzenie_kart(new boolean[] {false,false,false,false,false,false}, gneratorRozdan);
	
			if (i%10000==0)
				System.out.print("... "+i);
			
			assertEquals(wynik_java.length, ileWygranych);
			
			for (int j=0; j < wynik_java.length; j++)
				assertEquals(wynik_java[j], wynik_c[j]);
				
			gneratorRozdan.generate();
			ai_texas_swig.destruktorRozdanie( rozdanie );
		}			
		System.out.print("\nTest zakonczony sukcesem");		
	}	
	
}
