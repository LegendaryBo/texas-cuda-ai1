package cuda.test;

import engine.Hand;
import engine.RegulyGry;
import engine.TexasSettings;
import generator.GeneratorRozdan;
import junit.framework.TestCase;
import cuda.swig.SWIGTYPE_p_Hand;
import cuda.swig.ai_texas_swig;
import cuda.swig.texas_swig;

/**
 * Test funkcji iloscKartTejSamejWysokosci, napisane w C i w javie.
 * Testy porownawcze
 * 
 * @author Kacper Gorski (railman85@gmail.com)
 */
public class TestIleTakichSamychKart extends TestCase {

	final int ILOSC_SPRAWDZEN = 50000;

	public void setUp() {
		TexasSettings.setTexasLibraryPath();
	}
	
	public void testFunkcji() {
		
		int[] output_c = new int[2];
		byte[] output_java = new byte[2];
		
		System.out.println("\n\n==Test funkcji iloscKartTejSamejWysokosci ("+ILOSC_SPRAWDZEN+" testów) ==");
		System.out.println("liczba sprawdzeń... ");
		
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		for (int i=0; i < ILOSC_SPRAWDZEN; i++) {
			for (int j=0; j < 6; j++) {
				
				Hand hand = gneratorRozdan.getHand(j);
				output_java = RegulyGry.ile_kart_tej_same_wysokosci(hand.getKarty());
				
				SWIGTYPE_p_Hand wskaznik = hand.stworzObiektWSwigu();
				
				texas_swig.iloscKartTejSamejWysokosciHOST(wskaznik, output_c);
				
				assertTrue(output_java[0] <=5);
				assertTrue(output_java[0]>0);
				assertTrue(output_java[1] <=5);
				assertTrue(output_java[1]>0);				
				assertEquals(output_java[0], output_c[0]);
				assertEquals(output_java[1], output_c[1]);
				
				gneratorRozdan.generate();
				
				ai_texas_swig.destruktorHand(wskaznik);
			}
			
			if (i%10000==0)
				System.out.print("... "+i);
		}
		System.out.print(" Test zakonczono pomyslnie");
	}
}
