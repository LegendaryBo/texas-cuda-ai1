package cuda.test;

import java.util.Random;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.RegulaAbstrakcyjna;
import reguly.LicytacjaNaWejscie.RegulaBardzoWysokieKarty;
import reguly.LicytacjaNaWejscie.RegulaCzyKolorWRece;
import reguly.LicytacjaNaWejscie.RegulaCzyParaWRece;
import reguly.LicytacjaNaWejscie.RegulaOgraniczenieStawki;
import reguly.LicytacjaNaWejscie.RegulaWymaganychGlosow;
import reguly.LicytacjaNaWejscie.RegulaWysokieKarty;
import reguly.LicytacjaPozniej.RegulaJestRezultat;
import Gracze.GraczAIv3;
import Gracze.gracz_v3.GeneratorRegulv3;
import cuda.swig.SWIGTYPE_p_CzyGracR1;
import cuda.swig.SWIGTYPE_p_CzyGracRX;
import cuda.swig.SWIGTYPE_p_Gra;
import cuda.swig.SWIGTYPE_p_KodGraya;
import cuda.swig.SWIGTYPE_p_Reguly;
import cuda.swig.SWIGTYPE_p_int;
import cuda.swig.ai_texas_swig;
import cuda.swig.texas_swig;
import engine.Gra;
import engine.RegulyGry;
import engine.TexasSettings;
import engine.rezultaty.Rezultat;
import generator.GeneratorRozdan;
import generator.IndividualGenerator;

public class TestRegulCzyGrac extends TestCase {

	final int ILOSC_SPRAWDZEN = 30000;

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
	
		System.out.println("\n\n== Test regul CzyGrac ("+ILOSC_SPRAWDZEN+" testow) ==");
		System.out.print("sprawdzenie ");
		
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		Random random = new Random(1235);
		
		IndividualGenerator pIndividualGenerator = new IndividualGenerator(12, 2000);
		EvBinaryVectorIndividual[] pIndividual = new EvBinaryVectorIndividual[6];

		SWIGTYPE_p_int[] osobnikPointers = new SWIGTYPE_p_int[6];
		for (int j=0; j < 6; j++) {
			pIndividual[j] = pIndividualGenerator.generate();		
			osobnikPointers[j] = ai_texas_swig.getOsobnikPTR(pIndividual[j].getGenes(), 2000) ;
		}

		for (int i=0; i < ILOSC_SPRAWDZEN; i++) {
			int seed =  gneratorRozdan.getSeed( );
			int shift = Math.abs(seed);

			SWIGTYPE_p_Gra gra = texas_swig.getGraPTR();
			texas_swig.nowaGra(
					osobnikPointers[(6+shift%6)%6], 
					osobnikPointers[(7+shift%6)%6], 
					osobnikPointers[(8+shift%6)%6], 
					osobnikPointers[(9+shift%6)%6], 
					osobnikPointers[(10+shift%6)%6], 
					osobnikPointers[(11+shift%6)%6], 
					seed, 2, gra);					
			int index_start = random.nextInt(1900);
			int ktory_gracz = random.nextInt(6);
			float[] rezultat_c = new float[2];
			float rezultat_java;
			Gra gra_java = new Gra(gneratorRozdan);
			RegulaAbstrakcyjna regula_java;
			float[] wynik = new float[1];

			
		
			float stawka_gra = random.nextInt(2000);
			gra_java.stawka = stawka_gra;
			
			ai_texas_swig.setStawka(gra, (int)stawka_gra);
			
			
			SWIGTYPE_p_KodGraya kodGraya = ai_texas_swig.getKodGrayaPTR(index_start+1, 5);
			SWIGTYPE_p_KodGraya kodGraya2 = ai_texas_swig.getKodGrayaPTR(index_start+1+5, 5);
			
			/*		regula	grajGdyParaWRekuR1	 */
			ai_texas_swig.grajGdyParaWRekuR1HOST(gra, ktory_gracz, kodGraya, rezultat_c, stawka_gra);
			regula_java = new RegulaCzyParaWRece(index_start, pIndividual[ktory_gracz], DLUGOSC_KODU);
			rezultat_java = (float) regula_java.aplikujRegule(
					gra_java,
					ktory_gracz,
					pIndividual[ktory_gracz] ,
					RegulyGry.najlepsza_karta( gneratorRozdan.getAllCards(ktory_gracz) ));
			assertEquals(rezultat_java  , rezultat_c[0], 0.0001f);


			/*		regula	grajGdyParaWRekuR1	 */
			ai_texas_swig.grajGdyKolorWRekuR1HOST(gra, ktory_gracz, kodGraya, rezultat_c, stawka_gra);
			regula_java = new RegulaCzyKolorWRece(index_start, DLUGOSC_KODU);
			rezultat_java = (float) regula_java.aplikujRegule(
					gra_java,
					ktory_gracz,
					pIndividual[ktory_gracz] ,
					RegulyGry.najlepsza_karta( gneratorRozdan.getAllCards(ktory_gracz) ));
			assertEquals(rezultat_java  , rezultat_c[0], 0.0001f);			

			/*		regula	grajWysokieKartyNaWejscieR1	 */
			ai_texas_swig.grajWysokieKartyNaWejscieR1HOST(gra, ktory_gracz, kodGraya, rezultat_c, stawka_gra);
			regula_java = new RegulaWysokieKarty(index_start, DLUGOSC_KODU);
			rezultat_java = (float) regula_java.aplikujRegule(
					gra_java,
					ktory_gracz,
					pIndividual[ktory_gracz] ,
					RegulyGry.najlepsza_karta( gneratorRozdan.getAllCards(ktory_gracz) ));
			assertEquals(rezultat_java  , rezultat_c[0], 0.0001f);	
	
			
			/*		regula	grajBardzoWysokieKartyNaWejscieR1	 */
			ai_texas_swig.grajBardzoWysokieKartyNaWejscieR1HOST(gra, ktory_gracz, kodGraya, rezultat_c, stawka_gra);
			regula_java = new RegulaBardzoWysokieKarty(index_start, DLUGOSC_KODU);
			rezultat_java = (float) regula_java.aplikujRegule(
					gra_java,
					ktory_gracz,
					pIndividual[ktory_gracz] ,
					RegulyGry.najlepsza_karta( gneratorRozdan.getAllCards(ktory_gracz) ));
			assertEquals(rezultat_java  , rezultat_c[0], 0.0001f);				
	
			
			/*		regula	grajBardzoWysokieKartyNaWejscieR1	 */
			ai_texas_swig.wymaganychGlosowRXHOST(gra, ktory_gracz, kodGraya, rezultat_c);
			regula_java = new RegulaWymaganychGlosow(index_start+1, DLUGOSC_KODU);
			rezultat_java = (float) regula_java.aplikujRegule(
					gra_java,
					ktory_gracz,
					pIndividual[ktory_gracz] ,
					RegulyGry.najlepsza_karta( gneratorRozdan.getAllCards(ktory_gracz) ));
			assertEquals(rezultat_java  , rezultat_c[0], 0.0001f);				
			
			
			/*		regula	grajOgraniczenieStawkiNaWejscieR1	 */
			ai_texas_swig.grajOgraniczenieStawkiNaWejscieR1HOST(gra, ktory_gracz, kodGraya, kodGraya2, rezultat_c);
			regula_java = new RegulaOgraniczenieStawki(index_start, DLUGOSC_KODU, DLUGOSC_KODU2);
			rezultat_java = (float) regula_java.aplikujRegule(
					gra_java,
					ktory_gracz,
					pIndividual[ktory_gracz] ,
					RegulyGry.najlepsza_karta( gneratorRozdan.getAllCards(ktory_gracz) ));
			assertEquals(rezultat_java  , rezultat_c[0], 0.0001f);		
			
			
			/*		regula	grajBardzoWysokieKartyNaWejscieR1	 */
			ai_texas_swig.ograniczenieStawkiRXHOST(gra, ktory_gracz, kodGraya, kodGraya2, wynik);
			regula_java = new RegulaOgraniczenieStawki(index_start, DLUGOSC_KODU, DLUGOSC_KODU2);
			rezultat_java = (float) regula_java.aplikujRegule(
					gra_java,
					ktory_gracz,
					pIndividual[ktory_gracz] ,
					RegulyGry.najlepsza_karta( gneratorRozdan.getAllCards(ktory_gracz) ));
			
			assertEquals(rezultat_java  , wynik[0], 0.0001f);		
			
			
			
			final int RUNDA=4;
			gra_java.runda=RUNDA-1;
			int wymagany_rezultat=random.nextInt(5)+2;
			

			ai_texas_swig.setRunda(gra, 4);
			ai_texas_swig.grajRezultatRXHOST(gra, ktory_gracz, kodGraya, rezultat_c, wymagany_rezultat);
			regula_java = new RegulaJestRezultat(index_start, DLUGOSC_KODU, wymagany_rezultat);
			rezultat_java = (float) regula_java.aplikujRegule(
					gra_java,
					ktory_gracz,
					pIndividual[ktory_gracz] ,
					RegulyGry.najlepsza_karta( gneratorRozdan.getAllCards(ktory_gracz) ));

			assertEquals(rezultat_java  , rezultat_c[0], 0.0001f);	
			
			ai_texas_swig.destruktorGra(gra);
			
			if (i%5000==0)
				System.out.print("... "+i);
			
			gneratorRozdan.generate();
			ai_texas_swig.destruktorKodGraya(kodGraya);
			ai_texas_swig.destruktorKodGraya(kodGraya2);
			
		}
		for (int j = 0; j < osobnikPointers.length; j++) {
			ai_texas_swig.destruktorInt(osobnikPointers[j]);
		}
		
		System.out.print("\nTest zakonczony sukcesem");
	}
	
	
	
	
	public void testRegulRunda1() {
		
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		Random random = new Random(1235);
		
		System.out.println("\n\n== Test regul CzyGrac runda1 ("+ILOSC_SPRAWDZEN+" testow) ==");
		System.out.print("sprawdzenie ");
		
		IndividualGenerator pIndividualGenerator = new IndividualGenerator(12, 2000);
		EvBinaryVectorIndividual[] pIndividual = new EvBinaryVectorIndividual[6];

		for (int j=0; j < 6; j++) 
			pIndividual[j] = pIndividualGenerator.generate();		
		
		GeneratorRegulv3.init();
		
		SWIGTYPE_p_int[] osobnikPointers = new SWIGTYPE_p_int[6];
		for (int j=0; j < 6; j++) {
			pIndividual[j] = pIndividualGenerator.generate();		
			osobnikPointers[j] = ai_texas_swig.getOsobnikPTR(pIndividual[j].getGenes(), 2000) ;
		}	
		
		SWIGTYPE_p_Reguly reguly = ai_texas_swig.getReguly();
		SWIGTYPE_p_CzyGracR1  dobijanie = ai_texas_swig.getCzyGracR1PTRZReguly(reguly);
		
		for (int i=0; i < ILOSC_SPRAWDZEN; i++) {
			int seed = gneratorRozdan.getSeed();
			int shift = Math.abs(seed);
	
			if (i%100==0) {
				for (int j=0; j < 6; j++) {
					ai_texas_swig.destruktorInt(osobnikPointers[j]);
					pIndividual[j] = pIndividualGenerator.generate();		
					osobnikPointers[j] = ai_texas_swig.getOsobnikPTR(pIndividual[j].getGenes(), 2000) ;
				}	

			}
			
//			System.out.println("test nr"+i);

			SWIGTYPE_p_Gra gra = texas_swig.getGraPTR();
			
			texas_swig.nowaGra(
					osobnikPointers[(6+shift%6)%6], 
					osobnikPointers[(7+shift%6)%6], 
					osobnikPointers[(8+shift%6)%6], 
					osobnikPointers[(9+shift%6)%6], 
					osobnikPointers[(10+shift%6)%6], 
					osobnikPointers[(11+shift%6)%6], 
					seed, 2, gra);		
			
			int ktory_gracz = random.nextInt(6);
			
			float stawka_dummy  = (random.nextInt(1000)+6.0f)*5.0f;
			
			float[] wyniki = new float[1];
			

			ai_texas_swig.setStawka(gra, (int) stawka_dummy);
			ai_texas_swig.aplikujCzyGracR1HOST(dobijanie, gra, ktory_gracz, wyniki, stawka_dummy);	
			Gra gra_java = new Gra(gneratorRozdan);
			gra_java.stawka = stawka_dummy;
			GraczAIv3 gracz_ai_java = new GraczAIv3( pIndividual[ktory_gracz], ktory_gracz );
			gracz_ai_java.gra = gra_java;
			
			
			boolean wynik_java = gracz_ai_java.rundaX_czy_grac(1);
			float wynik_java_float;
			if (wynik_java)
				wynik_java_float=1.0f;
			else 
				wynik_java_float=0.0f;
			

			assertEquals(wynik_java_float, wyniki[0], 0.001);
		
			ai_texas_swig.destruktorGra(gra);
			
			if (i%5000==0)
				System.out.print("... "+i);
			
			gneratorRozdan.generate();

		}
		
		System.out.print("\nTest zakonczony sukcesem");
		
	}
	
	
	public void testRegulRunda2() {
		
		final int RUNDA=2;
		
		GeneratorRegulv3.init();
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		Random random = new Random(1235);
		
		System.out.println("\n\n== Test regul CzyGrac runda2 ("+ILOSC_SPRAWDZEN+" testow) ==");
		System.out.print("sprawdzenie ");
		
		IndividualGenerator pIndividualGenerator = new IndividualGenerator(12, 2000);
		EvBinaryVectorIndividual[] pIndividual = new EvBinaryVectorIndividual[6];

		SWIGTYPE_p_int[] osobnikPointers = new SWIGTYPE_p_int[6];
		for (int j=0; j < 6; j++) {
			pIndividual[j] = pIndividualGenerator.generate();		
			osobnikPointers[j] = ai_texas_swig.getOsobnikPTR(pIndividual[j].getGenes(), 2000) ;
		}
		

		SWIGTYPE_p_Reguly reguly = ai_texas_swig.getReguly();
		SWIGTYPE_p_CzyGracRX  czy_grac = ai_texas_swig.getCzyGracRXPTRZReguly(reguly, 2);
		
		for (int i=0; i < ILOSC_SPRAWDZEN; i++) {
			int seed =  gneratorRozdan.getSeed( );
			int shift = Math.abs(seed);

			if (i%100==0) {
				for (int j=0; j < 6; j++) {
					ai_texas_swig.destruktorInt(osobnikPointers[j]);
					pIndividual[j] = pIndividualGenerator.generate();		
					osobnikPointers[j] = ai_texas_swig.getOsobnikPTR(pIndividual[j].getGenes(), 2000) ;
				}	
			}
			
//			System.out.println("test nr"+i);

			SWIGTYPE_p_Gra gra = texas_swig.getGraPTR();
			
			
			texas_swig.nowaGra(
					osobnikPointers[(6+shift%6)%6], 
					osobnikPointers[(7+shift%6)%6], 
					osobnikPointers[(8+shift%6)%6], 
					osobnikPointers[(9+shift%6)%6], 
					osobnikPointers[(10+shift%6)%6], 
					osobnikPointers[(11+shift%6)%6], 
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
			
			boolean wynik_java = gracz_ai_java.rundaX_czy_grac(RUNDA);			
			
			
			ai_texas_swig.aplikujCzyGracRXHOST(czy_grac, gra, ktory_gracz, wyniki, stawka_dummy_proponowana);
			
//			System.out.println(wynik_java);
			assertEquals(wynik_java, wyniki[0]==1.0f);
			
			ai_texas_swig.destruktorGra(gra);
			
			if (i%5000==0)
				System.out.print("... "+i);
			
			gneratorRozdan.generate();
		}	
		
		for (int j = 0; j < osobnikPointers.length; j++) {
			ai_texas_swig.destruktorInt(osobnikPointers[j]);
		}
		
		System.out.print("\nTest zakonczony sukcesem");
	}	
	

	public void testRegulRunda3() {
		
		final int RUNDA=3;
		
		GeneratorRegulv3.init();
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		Random random = new Random(1235);
		
		System.out.println("\n\n== Test regul CzyGrac runda3 ("+ILOSC_SPRAWDZEN+" testow) ==");
		System.out.print("sprawdzenie ");
		
		IndividualGenerator pIndividualGenerator = new IndividualGenerator(12, 23000);
		EvBinaryVectorIndividual[] pIndividual = new EvBinaryVectorIndividual[6];

		
		
		SWIGTYPE_p_int[] osobnikPointers = new SWIGTYPE_p_int[6];
		for (int j=0; j < 6; j++) {
			pIndividual[j] = pIndividualGenerator.generate();		
			osobnikPointers[j] = ai_texas_swig.getOsobnikPTR(pIndividual[j].getGenes(), 2000) ;
		}	
		
		SWIGTYPE_p_Reguly reguly = ai_texas_swig.getReguly();
		SWIGTYPE_p_CzyGracRX  czy_grac = ai_texas_swig.getCzyGracRXPTRZReguly(reguly, 3);
		
		for (int i=0; i < ILOSC_SPRAWDZEN; i++) {
			int seed =  gneratorRozdan.getSeed( );
			int shift = Math.abs(seed);

			if (i%100==0) {
				for (int j=0; j < 6; j++) {
					ai_texas_swig.destruktorInt(osobnikPointers[j]);
					pIndividual[j] = pIndividualGenerator.generate();		
					osobnikPointers[j] = ai_texas_swig.getOsobnikPTR(pIndividual[j].getGenes(), 2000) ;
				}	
			}
			
//			System.out.println("test nr"+i);

			SWIGTYPE_p_Gra gra = texas_swig.getGraPTR();
			
			
			texas_swig.nowaGra(
					osobnikPointers[(6+shift%6)%6], 
					osobnikPointers[(7+shift%6)%6], 
					osobnikPointers[(8+shift%6)%6], 
					osobnikPointers[(9+shift%6)%6], 
					osobnikPointers[(10+shift%6)%6], 
					osobnikPointers[(11+shift%6)%6], 
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
			
			boolean wynik_java = gracz_ai_java.rundaX_czy_grac(RUNDA);			
			
			
			ai_texas_swig.aplikujCzyGracRXHOST(czy_grac, gra, ktory_gracz, wyniki, stawka_dummy_proponowana);
			
//			System.out.println(wynik_java);
			assertEquals(wynik_java, wyniki[0]==1.0f);
			
			ai_texas_swig.destruktorGra(gra);
			
			if (i%5000==0)
				System.out.print("... "+i);
			
			gneratorRozdan.generate();
		}
		
		System.out.print("\nTest zakonczony sukcesem");
	}		
	
	
	public void testRegulRunda4() {
		
		final int RUNDA=4;
		
		GeneratorRegulv3.init();
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		Random random = new Random(1235);
		
		System.out.println("\n\n== Test regul CzyGrac runda4 ("+ILOSC_SPRAWDZEN+" testow) ==");
		System.out.print("sprawdzenie ");
		
		IndividualGenerator pIndividualGenerator = new IndividualGenerator(12, 3000);
		EvBinaryVectorIndividual[] pIndividual = new EvBinaryVectorIndividual[6];

		SWIGTYPE_p_int[] osobnikPointers = new SWIGTYPE_p_int[6];
		for (int j=0; j < 6; j++) {
			pIndividual[j] = pIndividualGenerator.generate();		
			osobnikPointers[j] = ai_texas_swig.getOsobnikPTR(pIndividual[j].getGenes(), 2000) ;
		}	
		
		SWIGTYPE_p_Reguly reguly = ai_texas_swig.getReguly();
		SWIGTYPE_p_CzyGracRX  czy_grac = ai_texas_swig.getCzyGracRXPTRZReguly(reguly, 4);
		
		for (int i=0; i < ILOSC_SPRAWDZEN; i++) {
			int seed =  gneratorRozdan.getSeed( );
			int shift = Math.abs(seed);

			if (i%100==0) {
				for (int j=0; j < 6; j++) {
					ai_texas_swig.destruktorInt(osobnikPointers[j]);
					pIndividual[j] = pIndividualGenerator.generate();		
					osobnikPointers[j] = ai_texas_swig.getOsobnikPTR(pIndividual[j].getGenes(), 2000) ;
				}	
			}
			
//			System.out.println("test nr"+i);

			SWIGTYPE_p_Gra gra = texas_swig.getGraPTR();
			
			
			texas_swig.nowaGra(
					osobnikPointers[(6+shift%6)%6], 
					osobnikPointers[(7+shift%6)%6], 
					osobnikPointers[(8+shift%6)%6], 
					osobnikPointers[(9+shift%6)%6], 
					osobnikPointers[(10+shift%6)%6], 
					osobnikPointers[(11+shift%6)%6], 
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
			
			boolean wynik_java = gracz_ai_java.rundaX_czy_grac(RUNDA);			
			
			
			ai_texas_swig.aplikujCzyGracRXHOST(czy_grac, gra, ktory_gracz, wyniki, stawka_dummy_proponowana);
			
//			System.out.println(wynik_java);
			assertEquals(wynik_java, wyniki[0]==1.0f);
			
			ai_texas_swig.destruktorGra(gra);
			
			if (i%5000==0)
				System.out.print("... "+i);
			
			gneratorRozdan.generate();
		}
		
		
		for (int j = 0; j < osobnikPointers.length; j++) {
			ai_texas_swig.destruktorInt(osobnikPointers[j]);
		}
		
		System.out.print("\nTest zakonczony sukcesem");
		
	}	
	
	
	
}
