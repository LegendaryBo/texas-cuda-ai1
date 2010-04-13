package cuda.test;


import engine.Gra;
import engine.TexasSettings;
import generator.GeneratorRozdan;
import junit.framework.TestCase;
import cuda.swig.SWIGTYPE_p_Gra;
import cuda.swig.SWIGTYPE_p_int;
import cuda.swig.ai_texas_swig;
import cuda.swig.texas_swig;

/**
 * 
 * test pojedynczej rundy
 * 
 * @author railman
 *
 */
public class TestRozgrywka extends TestCase {

	final int ILOSC_SPRAWDZEN = 40000;

	public void setUp() {
		TexasSettings.setTexasLibraryPath();
	}
	
	public static int ile_porownan=0;
	
	/**
	 * Test losuje pewna ilosc rozdan i sprawdza, czy program w C i javie obliczyly tak samo
	 */
	public void atestRozrywkaStaliGracze() {

		final int typ_gracza=0;
		
		System.out.println("\n\n==Test rozgrywania partii, bez AI ("+ILOSC_SPRAWDZEN+" testow) ==");
		
		SWIGTYPE_p_int dummy = ai_texas_swig.getOsobnikPTR(new int[1], 1);
		
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		for (int i=0; i < ILOSC_SPRAWDZEN; i++) {
			int seed = gneratorRozdan.getSeed();
			
			SWIGTYPE_p_Gra gra = texas_swig.getGraPTR();
			System.out.println("huj");
			System.out.flush();
			texas_swig.nowaGra(dummy, dummy, dummy, dummy, dummy, dummy, seed, typ_gracza, gra);
			
			System.out.println("huj");
			System.out.flush();			
			float wynik_c =  texas_swig.rozegrajPartieHOST(gra, 4);
			Gra gra_java = new Gra(gneratorRozdan, typ_gracza);

			if (i%1==0)
				System.out.print("... "+i);
			gra_java.play_round(false);
//			System.out.println(gneratorRozdan);
//			System.out.println(gra_java.play_round(true));
//			
//			for (int j=0; j < 6; j++)
//				System.out.println("bilans "+j+" :" +gra_java.gracze[j].bilans );
			double wynik_java = gra_java.gracze[4].bilans;
			
			assertEquals((float)wynik_java, wynik_c);
	
			gneratorRozdan.generate();
			ai_texas_swig.destruktorGra(gra);
			
		}		
		System.out.print("\nTest zakonczony sukcesem");
	}		
	
	public void testDummy() {
		
	}
	
	public void atestRozrywkaRozniGracze() {

		final int typ_gracza=1;
		
		System.out.println("\n\n==Test rozgrywania partii, bez AI 2 ("+ILOSC_SPRAWDZEN+" testow)==");
		
		SWIGTYPE_p_int dummy = ai_texas_swig.getOsobnikPTR(new int[1], 1);
		
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		for (int i=0; i < ILOSC_SPRAWDZEN; i++) {
			int seed = gneratorRozdan.getSeed();
			
			SWIGTYPE_p_Gra gra = texas_swig.getGraPTR();
			
			texas_swig.nowaGra(dummy, dummy, dummy, dummy, dummy, dummy, seed, typ_gracza, gra);
			
			
			
			float wynik_c =  texas_swig.rozegrajPartieHOST(gra, 4);
			Gra gra_java = new Gra(gneratorRozdan, typ_gracza);

			if (i%10000==0)
				System.out.print("... "+i);
			gra_java.play_round(false);
//			System.out.println(gneratorRozdan);
//			System.out.println(gra_java.play_round(true));
//			
//			for (int j=0; j < 6; j++)
//				System.out.println("bilans "+j+" :" +gra_java.gracze[j].bilans );
			double wynik_java = gra_java.gracze[4].bilans;
			
			assertEquals((float)wynik_java, wynik_c);
	
			gneratorRozdan.generate();
			ai_texas_swig.destruktorGra(gra);
			
		}		
		System.out.print("\nTest zakonczony sukcesem");
	}			
	
}
