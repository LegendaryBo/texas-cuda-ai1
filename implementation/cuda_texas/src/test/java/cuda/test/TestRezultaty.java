package cuda.test;

import engine.Hand;
import engine.RegulyGry;
import engine.TexasSettings;
import generator.GeneratorRozdan;
import junit.framework.TestCase;
import cuda.swig.SWIGTYPE_p_Hand;
import cuda.swig.ai_texas_swig;
import cuda.swig.texas_swig;

public class TestRezultaty extends TestCase {

	final int ILOSC_SPRAWDZEN = 10000;

	public void setUp() {
		TexasSettings.setTexasLibraryPath();	
	}
	
	
	/**
	 * Test losuje pewna ilosc rozdan i sprawdza, czy program w C i javie obliczyly tak samo
	 */	
	public void testJestKolor() {
		
		int output_c;
		int output_java;
		int licznik_kolorow=0;
		
		System.out.println("\n\n==Test rozpoznawania koloru ("+ILOSC_SPRAWDZEN+" testow) ==");
		System.out.print("sprawdzenie ");
		
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		for (int i=0; i < ILOSC_SPRAWDZEN; i++) {
			for (int j=0; j < 6; j++) {
				
				Hand hand = gneratorRozdan.getHand(j);
				SWIGTYPE_p_Hand wskaznik = hand.stworzObiektWSwigu();
				
				output_c = texas_swig.jestKolorHOST(wskaznik);
				if (RegulyGry.jest_kolor(hand.getKarty()))
					output_java = 1;
				else
					output_java = 0;
				if (output_java==1) 
					licznik_kolorow++;	
				
				assertEquals(output_java, output_c);
				
				ai_texas_swig.destruktorHand(wskaznik);
					
			}
			gneratorRozdan.generate();
			if (i%2000==0)
				System.out.print("... "+i);
		}
		System.out.println("\nTest kolorow zdany, liczba kolorow "+licznik_kolorow);
		
	}	
	
	
	
	public void testJestStreet() {
		
		int output_c;
		int output_java;
		int licznik_streetow=0;
		
		System.out.println("\n\n==Test rozpoznawania streeta ("+ILOSC_SPRAWDZEN+" testow)==");
		System.out.print("sprawdzenie ");
		
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		for (int i=0; i < ILOSC_SPRAWDZEN; i++) {
			for (int j=0; j < 6; j++) {
				
				Hand hand = gneratorRozdan.getHand(j);
				SWIGTYPE_p_Hand wskaznik = hand.stworzObiektWSwigu();
					
				
				output_c = texas_swig.jestStreetHOST(wskaznik);
				if (RegulyGry.jest_street(hand.getKarty()))
					output_java = 1;
				else
					output_java = 0;
				if (output_java==1) 
					licznik_streetow++;
	
				assertEquals(output_java, output_c);
				
				gneratorRozdan.generate();
				ai_texas_swig.destruktorHand(wskaznik);
			}
			if (i%2000==0)
				System.out.print("... "+i);
		}
		
		System.out.println("\nTest streetow zdany, liczba streetow "+licznik_streetow);
	}		
	
	
	public void testJestPoker() {
	
		int output_c;
		int output_java;
		int licznik_pokerow=0;
		
		System.out.println("\n\n==Test rozpoznawania pokerow ("+ILOSC_SPRAWDZEN+" testow)==");
		System.out.print("sprawdzenie ");
		
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		for (int i=0; i < ILOSC_SPRAWDZEN; i++) {
			for (int j=0; j < 6; j++) {
				
				Hand hand = gneratorRozdan.getHand(j);
				SWIGTYPE_p_Hand wskaznik = hand.stworzObiektWSwigu();
					
				
				output_c = texas_swig.jestPokerHOST(wskaznik);
				if (RegulyGry.jest_poker(hand.getKarty()))
					output_java = 1;
				else
					output_java = 0;
				if (output_java==1) 
					licznik_pokerow++;
				
				
				assertEquals(output_java, output_c);
				
				gneratorRozdan.generate();
				ai_texas_swig.destruktorHand(wskaznik);
			}
			if (i%2000==0)
				System.out.print("... "+i);
		}		
		
		System.out.println("\nTest pokerow zdany, liczba pokerow "+licznik_pokerow);
		
	}
	
	
	/**
	 * Test losuje pewna ilosc rozdan i sprawdza, czy program w C i javie obliczyly tak samo
	 */
	public void testRezultatow() {
	
		int output_c;
		int output_java;
		int[] licznik = new int[10];
		
		System.out.println("\n\n==Test rozpoznawania handow ("+ILOSC_SPRAWDZEN+" testow)==");
		System.out.print("sprawdzenie ");
		
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		for (int i=0; i < ILOSC_SPRAWDZEN; i++) {
			for (int j=0; j < 6; j++) {
				
				Hand hand = gneratorRozdan.getHand(j);
				SWIGTYPE_p_Hand wskaznik = hand.stworzObiektWSwigu();
					
				
				output_c = texas_swig.najlepszaKartaHOST(wskaznik);
				output_java = RegulyGry.najlepsza_karta(hand.getKarty()).poziom;
				licznik[output_java]++;
				
				
				assertEquals(output_java, output_c);
				
				gneratorRozdan.generate();
				ai_texas_swig.destruktorHand(wskaznik);
			}
			if (i%2000==0)
				System.out.print("... "+i);
		}		
		
		int suma=0;
		for (int i=0; i < 10; i++)
			suma+=licznik[i];
		assertEquals(ILOSC_SPRAWDZEN*6, suma);
		
		System.out.println("Test rezultatow zdany, wyniki: ");		
		System.out.println("\tSmieciow - "+licznik[1]);
		System.out.println("\tPar - "+licznik[2]);
		System.out.println("\tDwoch Par - "+licznik[3]);
		System.out.println("\tTrojek - "+licznik[4]);
		System.out.println("\tStreetow - "+licznik[5]);
		System.out.println("\tKolorow - "+licznik[6]);
		System.out.println("\tFulli - "+licznik[7]);
		System.out.println("\tKaret - "+licznik[8]);		
		System.out.println("\tPokerow - "+licznik[9]);	
		
		System.out.print("\nTest zakonczony sukcesem");
	}
	
	
}
