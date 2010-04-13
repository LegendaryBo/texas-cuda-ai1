package cuda.test;

import engine.Gra;
import engine.RegulyGry;
import engine.TexasSettings;
import engine.rezultaty.Rezultat;
import generator.GeneratorRozdan;
import generator.IndividualGenerator;

import java.util.Random;

import Gracze.GraczAIv3;
import Gracze.gracz_v3.GeneratorRegulv3;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.RegulaAbstrakcyjnaIleGrac;
import reguly.ileGrac.IleGracBardzoWysokaKartaR1;
import reguly.ileGrac.IleGracKolorWRekuR1;
import reguly.ileGrac.IleGracParaWRekuR1;
import reguly.ileGrac.IleGracPulaRX;
import reguly.ileGrac.IleGracRezultatRX;
import reguly.ileGrac.IleGracStawkaRX;
import reguly.ileGrac.IleGracWysokaKartaWRekuR1;
import reguly.ileGrac.IleGracXGraczyWGrzeRX;
import cuda.swig.SWIGTYPE_p_Gra;
import cuda.swig.SWIGTYPE_p_IleGracR1;
import cuda.swig.SWIGTYPE_p_IleGracRX;
import cuda.swig.SWIGTYPE_p_KodGraya;
import cuda.swig.SWIGTYPE_p_Reguly;
import cuda.swig.SWIGTYPE_p_int;
import cuda.swig.ai_texas_swig;
import cuda.swig.texas_swig;

public class TestRegulIleGrac extends TestCase {

	final int ILOSC_SPRAWDZEN = 10000;

	public void setUp() {
		TexasSettings.setTexasLibraryPath();		
	}
	
	public static int ile_porownan=0;
	
	final int DLUGOSC_KODU=5;
	final int DLUGOSC_KODU2=5;
	
	/**
	 * Test losuje pewna ilosc rozdan i sprawdza, czy program w C i javie obliczyly tak samo
	 */
	public void testRegul() {
	
		System.out.println("\n\n==Test regul odpowiedzialnych za strategie obstawiania==");
		System.out.print("sprawdzenie ");
		
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		Random random = new Random(1235);
		
		IndividualGenerator pIndividualGenerator = new IndividualGenerator(12, 2000);
		EvBinaryVectorIndividual[] pIndividual = new EvBinaryVectorIndividual[6];

		for (int j=0; j < 6; j++) 
			pIndividual[j] = pIndividualGenerator.generate();		
		
		SWIGTYPE_p_int osobnik1 = ai_texas_swig.getOsobnikPTR(pIndividual[0].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik2 = ai_texas_swig.getOsobnikPTR(pIndividual[1].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik3 = ai_texas_swig.getOsobnikPTR(pIndividual[2].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik4 = ai_texas_swig.getOsobnikPTR(pIndividual[3].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik5 = ai_texas_swig.getOsobnikPTR(pIndividual[4].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik6 = ai_texas_swig.getOsobnikPTR(pIndividual[5].getGenes(), 2000) ;
		

		
		
		for (int i=0; i < ILOSC_SPRAWDZEN; i++) {
			int seed = gneratorRozdan.getSeed();


			SWIGTYPE_p_Gra gra = texas_swig.getGraPTR();
			
			
			texas_swig.nowaGra(
					osobnik1, 
					osobnik2, 
					osobnik3, 
					osobnik4, 
					osobnik5, 
					osobnik6, 
					seed, 2, gra);					
			int index_start = random.nextInt(1900);
			int ktory_gracz = random.nextInt(6);
			float[] rezultat_c = new float[2];
			float[] rezultat_java = null;
			RegulaAbstrakcyjnaIleGrac regula_java;

			SWIGTYPE_p_KodGraya kodGraya_waga = ai_texas_swig.getKodGrayaPTR(index_start+1, 5);
			SWIGTYPE_p_KodGraya kodGraya_jak_grac = ai_texas_swig.getKodGrayaPTR(index_start+1+5, DLUGOSC_KODU);
			
			
			/*		regula	IleGracParaWRekuR1	 */
			ai_texas_swig.ileGracParaWRekuR1HOST(gra, ktory_gracz, kodGraya_waga, kodGraya_jak_grac, rezultat_c);
			regula_java = new IleGracParaWRekuR1(index_start, DLUGOSC_KODU);
			rezultat_java = regula_java.aplikRegule(
					new Gra(gneratorRozdan),
					ktory_gracz,
					pIndividual[ktory_gracz] ,
					RegulyGry.najlepsza_karta( gneratorRozdan.getAllCards(ktory_gracz) ),
					0.0f);
			assertEquals(rezultat_java[0]  , rezultat_c[0], 0.0001f);
			assertEquals(rezultat_java[1]  , rezultat_c[1], 0.0001f);
			
			/*		regula	IleGracKolorWRekuR1	 */
			ai_texas_swig.ileGracKolorWRekuR1HOST(gra, ktory_gracz, kodGraya_waga, kodGraya_jak_grac, rezultat_c);
			regula_java = new IleGracKolorWRekuR1(index_start, DLUGOSC_KODU);
			rezultat_java = regula_java.aplikRegule(
					new Gra(gneratorRozdan),
					ktory_gracz,
					pIndividual[ktory_gracz] ,
					RegulyGry.najlepsza_karta( gneratorRozdan.getAllCards(ktory_gracz) ),
					0.0f);
			assertEquals(rezultat_java[0]  , rezultat_c[0], 0.0001f);
			assertEquals(rezultat_java[1]  , rezultat_c[1], 0.0001f);			
			
			/*		regula	IleGracWysokaKartaWRekuR1	 */
			ai_texas_swig.ileGracWysokaKartaWRekuR1HOST(gra, ktory_gracz, kodGraya_waga, kodGraya_jak_grac, rezultat_c);
			regula_java = new IleGracWysokaKartaWRekuR1(index_start, DLUGOSC_KODU);
			rezultat_java = regula_java.aplikRegule(
					new Gra(gneratorRozdan),
					ktory_gracz,
					pIndividual[ktory_gracz] ,
					RegulyGry.najlepsza_karta( gneratorRozdan.getAllCards(ktory_gracz) ),
					0.0f);
			assertEquals(rezultat_java[0]  , rezultat_c[0], 0.0001f);
			assertEquals(rezultat_java[1]  , rezultat_c[1], 0.0001f);						
			
			/*		regula	IleGracBardzoWysokaKartaWRekuR1	 */
			ai_texas_swig.ileGracBardzoWysokaKartaWRekuR1HOST(gra, ktory_gracz, kodGraya_waga, kodGraya_jak_grac, rezultat_c);
			regula_java = new IleGracBardzoWysokaKartaR1(index_start, DLUGOSC_KODU);
			rezultat_java = regula_java.aplikRegule(
					new Gra(gneratorRozdan),
					ktory_gracz,
					pIndividual[ktory_gracz] ,
					RegulyGry.najlepsza_karta( gneratorRozdan.getAllCards(ktory_gracz) ),
					0.0f);
			assertEquals(rezultat_java[0]  , rezultat_c[0], 0.0001f);
			assertEquals(rezultat_java[1]  , rezultat_c[1], 0.0001f);			
			
			/*		regula	IleGracBardzoWysokaKartaWRekuR1	 */
			int ile_przeciwnikow_w_regule =random.nextInt(5)+1;
			int ile_graczy_w_grze =random.nextInt(5)+2;
			Gra gra_java = new Gra(gneratorRozdan);
			gra_java.graczy_w_grze = ile_graczy_w_grze;
			ai_texas_swig.setIleGraczyWGrze(gra, ile_graczy_w_grze);
			ai_texas_swig.IleGracXGraczyWGrzeRXHOST(gra, ktory_gracz, kodGraya_waga, kodGraya_jak_grac, rezultat_c, ile_przeciwnikow_w_regule);
			regula_java = new IleGracXGraczyWGrzeRX(index_start, DLUGOSC_KODU, ile_przeciwnikow_w_regule);
			rezultat_java = regula_java.aplikRegule(
					gra_java,
					ktory_gracz,
					pIndividual[ktory_gracz] ,
					RegulyGry.najlepsza_karta( gneratorRozdan.getAllCards(ktory_gracz) ),
					0.0f);
			assertEquals(rezultat_java[0]  , rezultat_c[0], 0.0001f);
			assertEquals(rezultat_java[1]  , rezultat_c[1], 0.0001f);					
		
			/*		regula	IleGracPulaRX	 */			
			int pula_w_grze =(random.nextInt(100)+1 )*5;
			gra_java = new Gra(gneratorRozdan);
			gra_java.pula = pula_w_grze;
			ai_texas_swig.setPula(gra, pula_w_grze);
			SWIGTYPE_p_KodGraya kodGraya_pula = ai_texas_swig.getKodGrayaPTR(index_start+1+5+DLUGOSC_KODU, DLUGOSC_KODU2);
			ai_texas_swig.IleGracPulaRXHOST(gra, ktory_gracz, kodGraya_waga, kodGraya_jak_grac, kodGraya_pula, rezultat_c);
			regula_java = new IleGracPulaRX(index_start, DLUGOSC_KODU, DLUGOSC_KODU2);
			rezultat_java = regula_java.aplikRegule(
					gra_java,
					ktory_gracz,
					pIndividual[ktory_gracz] ,
					RegulyGry.najlepsza_karta( gneratorRozdan.getAllCards(ktory_gracz) ),
					0.0f);
			assertEquals(rezultat_java[0]  , rezultat_c[0], 0.0001f);
			assertEquals(rezultat_java[1]  , rezultat_c[1], 0.0001f);	
			
			
			/*		regula	IleGracStawkaRX	 */			
			int stawka_w_grze =(random.nextInt(100)+1 )*5;
			gra_java = new Gra(gneratorRozdan);
			gra_java.stawka = stawka_w_grze;
			ai_texas_swig.setStawka(gra, stawka_w_grze);
			kodGraya_pula = ai_texas_swig.getKodGrayaPTR(index_start+1+5+DLUGOSC_KODU, DLUGOSC_KODU2);
			ai_texas_swig.IleGracStawkaRXHOST(gra, ktory_gracz, kodGraya_waga, kodGraya_jak_grac, kodGraya_pula, rezultat_c);
			regula_java = new IleGracStawkaRX(index_start, DLUGOSC_KODU, DLUGOSC_KODU2);
			rezultat_java = regula_java.aplikRegule(
					gra_java,
					ktory_gracz,
					pIndividual[ktory_gracz] ,
					RegulyGry.najlepsza_karta( gneratorRozdan.getAllCards(ktory_gracz) ),
					0.0f);
			assertEquals(rezultat_java[0]  , rezultat_c[0], 0.0001f);
			assertEquals(rezultat_java[1]  , rezultat_c[1], 0.0001f);	
			
			/*		regula	IleGracRezultatRX	 */			
			int ktora_runda = random.nextInt(3)+2;
			int jaki_rezultat = random.nextInt(8)+1;
			gra_java = new Gra(gneratorRozdan);
			gra_java.runda = ktora_runda-1;
			Rezultat rez_java =  Rezultat.pobierzPrognoze(gra_java, ktory_gracz);

			ai_texas_swig.setRunda(gra, ktora_runda);
			ai_texas_swig.IleGracRezultatRXHOST(gra, ktory_gracz, kodGraya_waga, kodGraya_jak_grac, rezultat_c,jaki_rezultat);
			regula_java = new IleGracRezultatRX(index_start, DLUGOSC_KODU, jaki_rezultat);
			rezultat_java = regula_java.aplikRegule(
					gra_java,
					ktory_gracz,
					pIndividual[ktory_gracz] ,
					rez_java,
					0.0f);
			assertEquals(rezultat_java[0]  , rezultat_c[0], 0.0001f);
			assertEquals(rezultat_java[1]  , rezultat_c[1], 0.0001f);				
			
			
			
			if (i%5000==0)
				System.out.println("... "+i);
			
			gneratorRozdan.generate();
			
			ai_texas_swig.destruktorGra(gra);
			ai_texas_swig.destruktorKodGraya(kodGraya_pula);
			ai_texas_swig.destruktorKodGraya(kodGraya_jak_grac);
			ai_texas_swig.destruktorKodGraya(kodGraya_waga);
		}	
		System.out.print("\nTest zakonczony sukcesem");
	}		
	
	
	public void testRegulRunda1() {
		
		
		System.out.println("\n\n==Test regul odpowiedzialnych za strategie obstawiania, runda1==");
		System.out.print("sprawdzenie ");
		
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		Random random = new Random(1235);
		
		IndividualGenerator pIndividualGenerator = new IndividualGenerator(12, 2000);
		EvBinaryVectorIndividual[] pIndividual = new EvBinaryVectorIndividual[6];

		for (int j=0; j < 6; j++) 
			pIndividual[j] = pIndividualGenerator.generate();		
		

		SWIGTYPE_p_Reguly reguly = ai_texas_swig.getReguly();
		SWIGTYPE_p_IleGracR1  ile_grac = ai_texas_swig.getIleGracR1PTRZReguly(reguly);
		
		GeneratorRegulv3.init();
		
		SWIGTYPE_p_int osobnik1 = ai_texas_swig.getOsobnikPTR(pIndividual[0].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik2 = ai_texas_swig.getOsobnikPTR(pIndividual[1].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik3 = ai_texas_swig.getOsobnikPTR(pIndividual[2].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik4 = ai_texas_swig.getOsobnikPTR(pIndividual[3].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik5 = ai_texas_swig.getOsobnikPTR(pIndividual[4].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik6 = ai_texas_swig.getOsobnikPTR(pIndividual[5].getGenes(), 2000) ;		
		
		for (int i=0; i < ILOSC_SPRAWDZEN; i++) {
			int seed = gneratorRozdan.getSeed();
	
			if (i%100==0) {
				
				ai_texas_swig.destruktorInt(osobnik1);
				ai_texas_swig.destruktorInt(osobnik2);
				ai_texas_swig.destruktorInt(osobnik3);
				ai_texas_swig.destruktorInt(osobnik4);
				ai_texas_swig.destruktorInt(osobnik5);
				ai_texas_swig.destruktorInt(osobnik6);
				for (int j=0; j < 6; j++) 
					pIndividual[j] = pIndividualGenerator.generate();
				osobnik1 = ai_texas_swig.getOsobnikPTR(pIndividual[0].getGenes(), 2000) ;
				osobnik2 = ai_texas_swig.getOsobnikPTR(pIndividual[1].getGenes(), 2000) ;
				osobnik3 = ai_texas_swig.getOsobnikPTR(pIndividual[2].getGenes(), 2000) ;
				osobnik4 = ai_texas_swig.getOsobnikPTR(pIndividual[3].getGenes(), 2000) ;
				osobnik5 = ai_texas_swig.getOsobnikPTR(pIndividual[4].getGenes(), 2000) ;
				osobnik6 = ai_texas_swig.getOsobnikPTR(pIndividual[5].getGenes(), 2000) ;
			}
			
//			System.out.println("test nr"+i);

			SWIGTYPE_p_Gra gra = texas_swig.getGraPTR();
			
			
			texas_swig.nowaGra(
					osobnik1, 
					osobnik2, 
					osobnik3, 
					osobnik4, 
					osobnik5, 
					osobnik6, 
					seed, 2, gra);		

			int ktory_gracz = random.nextInt(6);
			
			float stawka_dummy  = (random.nextInt(1000)+6.0f)*5.0f;
			
			float[] wyniki = new float[1];
			
			
			ai_texas_swig.aplikujIleGracR1HOST(ile_grac, gra, ktory_gracz, wyniki, stawka_dummy);
			
			Gra gra_java = new Gra(gneratorRozdan);
			
			GraczAIv3 gracz_ai_java = new GraczAIv3( pIndividual[ktory_gracz], ktory_gracz );
			double wynik_java = gracz_ai_java.rundaX_ileGrac(stawka_dummy , gra_java, 1);
			
			assertEquals(wynik_java, wyniki[0], 0.001);
			
			if (i%5000==0)
				System.out.println("... "+i);
			
			ai_texas_swig.destruktorGra(gra);
		}
		
		ai_texas_swig.destruktorInt(osobnik1);
		ai_texas_swig.destruktorInt(osobnik2);
		ai_texas_swig.destruktorInt(osobnik3);
		ai_texas_swig.destruktorInt(osobnik4);
		ai_texas_swig.destruktorInt(osobnik5);
		ai_texas_swig.destruktorInt(osobnik6);
		System.out.print("\nTest zakonczony sukcesem");
	}
	
	
	public void testRegulRunda2() {
		
		System.out.println("\n\n==Test regul odpowiedzialnych za strategie obstawiania, runda2==");
		System.out.print("sprawdzenie ");
		
		final int RUNDA=2;
		
		GeneratorRegulv3.init();
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		Random random = new Random(1235);
		
		IndividualGenerator pIndividualGenerator = new IndividualGenerator(12, 2000);
		EvBinaryVectorIndividual[] pIndividual = new EvBinaryVectorIndividual[6];

		for (int j=0; j < 6; j++) 
			pIndividual[j] = pIndividualGenerator.generate();		
		
		SWIGTYPE_p_int osobnik1 = ai_texas_swig.getOsobnikPTR(pIndividual[0].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik2 = ai_texas_swig.getOsobnikPTR(pIndividual[1].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik3 = ai_texas_swig.getOsobnikPTR(pIndividual[2].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik4 = ai_texas_swig.getOsobnikPTR(pIndividual[3].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik5 = ai_texas_swig.getOsobnikPTR(pIndividual[4].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik6 = ai_texas_swig.getOsobnikPTR(pIndividual[5].getGenes(), 2000) ;		
		
		SWIGTYPE_p_Reguly reguly = ai_texas_swig.getReguly();
		SWIGTYPE_p_IleGracRX  ile_grac = ai_texas_swig.getIleGracRXPTRZReguly(reguly, 2);
		
		
		for (int i=0; i < ILOSC_SPRAWDZEN; i++) {
			int seed = gneratorRozdan.getSeed();

			if (i%100==0) {
				ai_texas_swig.destruktorInt(osobnik1);
				ai_texas_swig.destruktorInt(osobnik2);
				ai_texas_swig.destruktorInt(osobnik3);
				ai_texas_swig.destruktorInt(osobnik4);
				ai_texas_swig.destruktorInt(osobnik5);
				ai_texas_swig.destruktorInt(osobnik6);
				
				for (int j=0; j < 6; j++) 
					pIndividual[j] = pIndividualGenerator.generate();
				osobnik1 = ai_texas_swig.getOsobnikPTR(pIndividual[0].getGenes(), 2000) ;
				osobnik2 = ai_texas_swig.getOsobnikPTR(pIndividual[1].getGenes(), 2000) ;
				osobnik3 = ai_texas_swig.getOsobnikPTR(pIndividual[2].getGenes(), 2000) ;
				osobnik4 = ai_texas_swig.getOsobnikPTR(pIndividual[3].getGenes(), 2000) ;
				osobnik5 = ai_texas_swig.getOsobnikPTR(pIndividual[4].getGenes(), 2000) ;
				osobnik6 = ai_texas_swig.getOsobnikPTR(pIndividual[5].getGenes(), 2000) ;
			}
			
//			System.out.println("test nr"+i);

			SWIGTYPE_p_Gra gra = texas_swig.getGraPTR();
			
			
			texas_swig.nowaGra(
					osobnik1, 
					osobnik2, 
					osobnik3, 
					osobnik4, 
					osobnik5, 
					osobnik6, 
					seed, 2, gra);	
			
			int ktory_gracz = random.nextInt(6);
			
			float stawka_dummy  = (random.nextInt(1000)+6.0f)*5.0f;
			float stawka_dummy_proponowana  = (random.nextInt(1000)+6.0f)*5.0f;
			ai_texas_swig.setStawka(gra, (int) stawka_dummy);
			
			float[] wyniki = new float[1];
			ai_texas_swig.setRunda(gra, RUNDA);
			Gra gra_java = new Gra(gneratorRozdan);
			GraczAIv3 gracz_ai_java = new GraczAIv3( pIndividual[ktory_gracz], ktory_gracz );
			gra_java.runda=RUNDA-1;
			gra_java.stawka=stawka_dummy;
			gracz_ai_java.rezultat = Rezultat.pobierzPrognoze(gra_java, ktory_gracz);
			gracz_ai_java.gra = gra_java;
			
			double wynik_java = gracz_ai_java.rundaX_ileGrac(stawka_dummy_proponowana,gra_java, RUNDA);			
			
			ai_texas_swig.aplikujIleGracRXHOST(ile_grac, gra, ktory_gracz, wyniki, stawka_dummy_proponowana);
			
			assertEquals(wynik_java, wyniki[0], 0.001f);
			
			if (i%5000==0)
				System.out.println("... "+i);
			
			ai_texas_swig.destruktorGra(gra);
			
			gneratorRozdan.generate();
		}
		
		System.out.print("\nTest zakonczony sukcesem");
	}	
	
	

	
	public void testRegulRunda3() {
		
		System.out.println("\n\n==Test regul odpowiedzialnych za strategie obstawiania, runda3==");
		System.out.print("sprawdzenie ");
		
		final int RUNDA=3;
		
		GeneratorRegulv3.init();
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		Random random = new Random(1235);
		
		IndividualGenerator pIndividualGenerator = new IndividualGenerator(12, 2000);
		EvBinaryVectorIndividual[] pIndividual = new EvBinaryVectorIndividual[6];

		for (int j=0; j < 6; j++) 
			pIndividual[j] = pIndividualGenerator.generate();		
		
		SWIGTYPE_p_int osobnik1 = ai_texas_swig.getOsobnikPTR(pIndividual[0].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik2 = ai_texas_swig.getOsobnikPTR(pIndividual[1].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik3 = ai_texas_swig.getOsobnikPTR(pIndividual[2].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik4 = ai_texas_swig.getOsobnikPTR(pIndividual[3].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik5 = ai_texas_swig.getOsobnikPTR(pIndividual[4].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik6 = ai_texas_swig.getOsobnikPTR(pIndividual[5].getGenes(), 2000) ;		
		
		SWIGTYPE_p_Reguly reguly = ai_texas_swig.getReguly();
		SWIGTYPE_p_IleGracRX  ile_grac = ai_texas_swig.getIleGracRXPTRZReguly(reguly, 3);
		
		for (int i=0; i < ILOSC_SPRAWDZEN; i++) {
			int seed = gneratorRozdan.getSeed();

			if (i%100==0) {
				
				ai_texas_swig.destruktorInt(osobnik1);
				ai_texas_swig.destruktorInt(osobnik2);
				ai_texas_swig.destruktorInt(osobnik3);
				ai_texas_swig.destruktorInt(osobnik4);
				ai_texas_swig.destruktorInt(osobnik5);
				ai_texas_swig.destruktorInt(osobnik6);				
				
				for (int j=0; j < 6; j++) 
					pIndividual[j] = pIndividualGenerator.generate();
				osobnik1 = ai_texas_swig.getOsobnikPTR(pIndividual[0].getGenes(), 2000) ;
				osobnik2 = ai_texas_swig.getOsobnikPTR(pIndividual[1].getGenes(), 2000) ;
				osobnik3 = ai_texas_swig.getOsobnikPTR(pIndividual[2].getGenes(), 2000) ;
				osobnik4 = ai_texas_swig.getOsobnikPTR(pIndividual[3].getGenes(), 2000) ;
				osobnik5 = ai_texas_swig.getOsobnikPTR(pIndividual[4].getGenes(), 2000) ;
				osobnik6 = ai_texas_swig.getOsobnikPTR(pIndividual[5].getGenes(), 2000) ;
			}
			
//			System.out.println("test nr"+i);

			SWIGTYPE_p_Gra gra = texas_swig.getGraPTR();
			
			
			texas_swig.nowaGra(
					osobnik1, 
					osobnik2, 
					osobnik3, 
					osobnik4, 
					osobnik5, 
					osobnik6, 
					seed, 2, gra);	
			

			
			int ktory_gracz = random.nextInt(6);
			
			float stawka_dummy  = (random.nextInt(1000)+6.0f)*5.0f;
			float stawka_dummy_proponowana  = (random.nextInt(1000)+6.0f)*5.0f;
			ai_texas_swig.setStawka(gra, (int) stawka_dummy);
			
			float[] wyniki = new float[1];
			ai_texas_swig.setRunda(gra, RUNDA);
			Gra gra_java = new Gra(gneratorRozdan);
			GraczAIv3 gracz_ai_java = new GraczAIv3( pIndividual[ktory_gracz], ktory_gracz );
			gra_java.runda=RUNDA-1;
			gra_java.stawka=stawka_dummy;
			gracz_ai_java.rezultat = Rezultat.pobierzPrognoze(gra_java, ktory_gracz);
			gracz_ai_java.gra = gra_java;
			
			double wynik_java = gracz_ai_java.rundaX_ileGrac(stawka_dummy_proponowana,gra_java, RUNDA);			
			
			
			ai_texas_swig.aplikujIleGracRXHOST(ile_grac, gra, ktory_gracz, wyniki, stawka_dummy_proponowana);
			
			assertEquals(wynik_java, wyniki[0], 0.001f);
			
			if (i%5000==0)
				System.out.println("... "+i);
			
			gneratorRozdan.generate();
			ai_texas_swig.destruktorGra(gra);
		}
		
		System.out.print("\nTest zakonczony sukcesem");
	}	
	
	
	public void testRegulRunda4() {
		
		System.out.println("\n\n==Test regul odpowiedzialnych za strategie obstawiania, runda4==");
		System.out.print("sprawdzenie ");
		
		final int RUNDA=4;
		
		GeneratorRegulv3.init();
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		Random random = new Random(1235);
		
		IndividualGenerator pIndividualGenerator = new IndividualGenerator(12, 3000);
		EvBinaryVectorIndividual[] pIndividual = new EvBinaryVectorIndividual[6];

		for (int j=0; j < 6; j++) 
			pIndividual[j] = pIndividualGenerator.generate();		
		
		SWIGTYPE_p_int osobnik1 = ai_texas_swig.getOsobnikPTR(pIndividual[0].getGenes(), 3000) ;
		SWIGTYPE_p_int osobnik2 = ai_texas_swig.getOsobnikPTR(pIndividual[1].getGenes(), 3000) ;
		SWIGTYPE_p_int osobnik3 = ai_texas_swig.getOsobnikPTR(pIndividual[2].getGenes(), 3000) ;
		SWIGTYPE_p_int osobnik4 = ai_texas_swig.getOsobnikPTR(pIndividual[3].getGenes(), 3000) ;
		SWIGTYPE_p_int osobnik5 = ai_texas_swig.getOsobnikPTR(pIndividual[4].getGenes(), 3000) ;
		SWIGTYPE_p_int osobnik6 = ai_texas_swig.getOsobnikPTR(pIndividual[5].getGenes(), 2000) ;		
		
		SWIGTYPE_p_Reguly reguly = ai_texas_swig.getReguly();
		SWIGTYPE_p_IleGracRX  ile_grac = ai_texas_swig.getIleGracRXPTRZReguly(reguly, 4);
		
		for (int i=0; i < ILOSC_SPRAWDZEN; i++) {
			int seed = gneratorRozdan.getSeed();

			if (i%100==0) {
				
				ai_texas_swig.destruktorInt(osobnik1);
				ai_texas_swig.destruktorInt(osobnik2);
				ai_texas_swig.destruktorInt(osobnik3);
				ai_texas_swig.destruktorInt(osobnik4);
				ai_texas_swig.destruktorInt(osobnik5);
				ai_texas_swig.destruktorInt(osobnik6);				
				
				for (int j=0; j < 6; j++) 
					pIndividual[j] = pIndividualGenerator.generate();
				osobnik1 = ai_texas_swig.getOsobnikPTR(pIndividual[0].getGenes(), 3000) ;
				osobnik2 = ai_texas_swig.getOsobnikPTR(pIndividual[1].getGenes(), 3000) ;
				osobnik3 = ai_texas_swig.getOsobnikPTR(pIndividual[2].getGenes(), 3000) ;
				osobnik4 = ai_texas_swig.getOsobnikPTR(pIndividual[3].getGenes(), 3000) ;
				osobnik5 = ai_texas_swig.getOsobnikPTR(pIndividual[4].getGenes(), 3000) ;
				osobnik6 = ai_texas_swig.getOsobnikPTR(pIndividual[5].getGenes(), 3000) ;
			}
			
//			System.out.println("test nr"+i);

			SWIGTYPE_p_Gra gra = texas_swig.getGraPTR();
			
			
			texas_swig.nowaGra(
					osobnik1, 
					osobnik2, 
					osobnik3, 
					osobnik4, 
					osobnik5, 
					osobnik6, 
					seed, 2, gra);	
			

			
			int ktory_gracz = random.nextInt(6);
			
			float stawka_dummy  = (random.nextInt(1000)+6.0f)*5.0f;
			float stawka_dummy_proponowana  = (random.nextInt(1000)+6.0f)*5.0f;
			ai_texas_swig.setStawka(gra, (int) stawka_dummy);
			
			float[] wyniki = new float[1];
			ai_texas_swig.setRunda(gra, RUNDA);
			Gra gra_java = new Gra(gneratorRozdan);
			GraczAIv3 gracz_ai_java = new GraczAIv3( pIndividual[ktory_gracz], ktory_gracz );
			gra_java.runda=RUNDA-1;
			gra_java.stawka=stawka_dummy;
			gracz_ai_java.rezultat = Rezultat.pobierzPrognoze(gra_java, ktory_gracz);
			gracz_ai_java.gra = gra_java;
			
			double wynik_java = gracz_ai_java.rundaX_ileGrac(stawka_dummy_proponowana,gra_java, RUNDA);			
			
			
			ai_texas_swig.aplikujIleGracRXHOST(ile_grac, gra, ktory_gracz, wyniki, stawka_dummy_proponowana);
			
			assertEquals(wynik_java, wyniki[0], 0.001f);
			
			if (i%5000==0)
				System.out.println("... "+i);
			
			gneratorRozdan.generate();
			ai_texas_swig.destruktorGra(gra);
		}
		
		System.out.print("\nTest zakonczony sukcesem");
	}	
	
	
	
	
}
