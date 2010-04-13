package cuda.test;

import engine.Gra;
import engine.TexasSettings;
import engine.rezultaty.Rezultat;
import generator.GeneratorRozdan;
import generator.IndividualGenerator;

import java.util.Random;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.RegulaAbstrakcyjnaDobijania;
import reguly.dobijanie.RegulaDobijacZawsze;
import reguly.dobijanie.RegulaDobijajGdyBrakujeX;
import reguly.dobijanie.RegulaDobijajGdyDobraKarta;
import reguly.dobijanie.RegulaDobijajGdyParaWRece;
import reguly.dobijanie.RegulaDobijajGdyWysokaReka;
import Gracze.GraczAIv3;
import Gracze.gracz_v3.GeneratorRegulv3;
import cuda.swig.SWIGTYPE_p_DobijanieR1;
import cuda.swig.SWIGTYPE_p_DobijanieRX;
import cuda.swig.SWIGTYPE_p_Gra;
import cuda.swig.SWIGTYPE_p_KodGraya;
import cuda.swig.SWIGTYPE_p_Reguly;
import cuda.swig.SWIGTYPE_p_int;
import cuda.swig.ai_texas_swig;
import cuda.swig.texas_swig;

public class TestRegulyDobijania extends TestCase {

	final int ILOSC_SPRAWDZEN = 30000;

	public void setUp() {
		TexasSettings.setTexasLibraryPath();	
	}
	
	public static int ile_porownan=0;
	
	final int DLUGOSC_KODU=5;	
	
	/**
	 * Test losuje pewna ilosc rozdan i sprawdza, czy program w C i javie obliczyly tak samo
	 */
	public void testRegul() {
	
		System.out.println("\n\n==Test regula odpowiedzialnych za dobijanie==");
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
//			float[] rezultat_c = new float[2];
//			float[] rezultat_java = null;
			Gra gra_java = new Gra(gneratorRozdan);
			RegulaAbstrakcyjnaDobijania regula_java;
			float[] wynik = new float[1];
			double wspolczynnik = 0.7d;
			
			float stawka = random.nextInt(2000);
			float stawka_gra = random.nextInt(2000);
			gra_java.stawka = stawka_gra;
			ai_texas_swig.setStawka(gra, (int)stawka_gra);
			
			
			SWIGTYPE_p_KodGraya kodGraya = ai_texas_swig.getKodGrayaPTR(index_start+1, 5);
			
			ai_texas_swig.dobijajZawszeHOST(gra, ktory_gracz, index_start, wynik);
			regula_java = new RegulaDobijacZawsze(index_start);
			double wynik_java = regula_java.aplikujRegule(gra_java, ktory_gracz, stawka, pIndividual[ktory_gracz], null);
			assertEquals((float)wynik_java, wynik[0]);

			ai_texas_swig.dobijajGdyParaWRekuR1HOST(gra, ktory_gracz, index_start, wynik, stawka, (float)wspolczynnik);
			regula_java = new RegulaDobijajGdyParaWRece(index_start, wspolczynnik);
			wynik_java = regula_java.aplikujRegule(gra_java, ktory_gracz, stawka, pIndividual[ktory_gracz], null);
			assertEquals((float)wynik_java, wynik[0]);
			
			ai_texas_swig.dobijajGdyWysokaKartaR1HOST(gra, ktory_gracz, index_start, wynik, stawka, (float)wspolczynnik);
			regula_java = new RegulaDobijajGdyWysokaReka(index_start, wspolczynnik);
			wynik_java = regula_java.aplikujRegule(gra_java, ktory_gracz, stawka, pIndividual[ktory_gracz], null);
			assertEquals((float)wynik_java, wynik[0]);
	
			int wymagany_rezultat = random.nextInt(5)+2;
			int runda = random.nextInt(3)+2;
			ai_texas_swig.setRunda(gra, runda);
			ai_texas_swig.dobijajGdyWysokaKartaRXHOST(gra, ktory_gracz, index_start, wynik, stawka, wymagany_rezultat);
			gra_java.runda=runda-1;
			regula_java = new RegulaDobijajGdyDobraKarta(index_start, wymagany_rezultat);
			wynik_java = regula_java.aplikujRegule(gra_java, ktory_gracz, stawka, pIndividual[ktory_gracz], 
					Rezultat.pobierzPrognoze(gra_java, ktory_gracz));			
			assertEquals((float)wynik_java, wynik[0]);			
			
			ai_texas_swig.dobijajGdyBrakujeXRXHOST(gra, ktory_gracz, index_start, wynik, stawka, (float)wspolczynnik);
			regula_java = new RegulaDobijajGdyBrakujeX(index_start, wspolczynnik);
			wynik_java = regula_java.aplikujRegule(gra_java, ktory_gracz, stawka, pIndividual[ktory_gracz], null);
			assertEquals((float)wynik_java, wynik[0]);			
			
			
			gneratorRozdan.generate();
			
			ai_texas_swig.destruktorGra(gra);
			ai_texas_swig.destruktorKodGraya(kodGraya);
			
			
			if (i%5000==0)
				System.out.print("... "+i);
		}
		System.out.print("\nTest zakonczony sukcesem");
	}
	
	
	
	
	public void testRegulRunda1() {
		
		System.out.println("\n\n==Test regula odpowiedzialnych za dobijanie runda 1==");
		System.out.print("sprawdzenie ");	
		
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		Random random = new Random(1235);
		
		IndividualGenerator pIndividualGenerator = new IndividualGenerator(12, 2000);
		EvBinaryVectorIndividual[] pIndividual = new EvBinaryVectorIndividual[6];

		for (int j=0; j < 6; j++) 
			pIndividual[j] = pIndividualGenerator.generate();		
		
		GeneratorRegulv3.init();
		
		SWIGTYPE_p_int osobnik1 = ai_texas_swig.getOsobnikPTR(pIndividual[0].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik2 = ai_texas_swig.getOsobnikPTR(pIndividual[1].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik3 = ai_texas_swig.getOsobnikPTR(pIndividual[2].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik4 = ai_texas_swig.getOsobnikPTR(pIndividual[3].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik5 = ai_texas_swig.getOsobnikPTR(pIndividual[4].getGenes(), 2000) ;
		SWIGTYPE_p_int osobnik6 = ai_texas_swig.getOsobnikPTR(pIndividual[5].getGenes(), 2000) ;		
		
//		int przesuniecie = 106;
		SWIGTYPE_p_Reguly reguly = ai_texas_swig.getReguly();
		SWIGTYPE_p_DobijanieR1  dobijanie = ai_texas_swig.getDobijanieR1PTRZReguly(reguly);
		
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
			
			
			ai_texas_swig.aplikujDobijanieR1HOST(dobijanie, gra, ktory_gracz, wyniki, stawka_dummy);	
			Gra gra_java = new Gra(gneratorRozdan);
			GraczAIv3 gracz_ai_java = new GraczAIv3( pIndividual[ktory_gracz], ktory_gracz );
			gracz_ai_java.gra = gra_java;
			
			boolean wynik_java = gracz_ai_java.rundaX_dobijanie(stawka_dummy, 1);
			float wynik_java_float;
			if (wynik_java)
				wynik_java_float=1.0f;
			else 
				wynik_java_float=0.0f;
			
			
			assertEquals(wynik_java_float, wyniki[0], 0.001);
		
			
			if (i%5000==0)
				System.out.print("... "+i);
			ai_texas_swig.destruktorGra(gra);
		}
		
		System.out.print("\nTest zakonczony sukcesem");
	}

	
	public void testRegulRunda2() {
		
		final int RUNDA=2;
		
		System.out.println("\n\n==Test regula odpowiedzialnych za dobijanie runda 2==");
		System.out.print("sprawdzenie ");	
		
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
		SWIGTYPE_p_DobijanieRX  ile_grac = ai_texas_swig.getDobijanieRXPTRZReguly(reguly, 2);
		
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
			
			boolean wynik_java = gracz_ai_java.rundaX_dobijanie(stawka_dummy_proponowana, RUNDA);			
			
			
			ai_texas_swig.aplikujDobijanieRXHOST(ile_grac, gra, ktory_gracz, wyniki, stawka_dummy_proponowana);
			

			assertEquals(wynik_java, wyniki[0]==1.0f);
			
			if (i%5000==0)
				System.out.print("... "+i);
			
			gneratorRozdan.generate();
			ai_texas_swig.destruktorGra(gra);
		}
		
		System.out.print("\nTest zakonczony sukcesem");
	}		
	
	
	
	public void testRegulRunda3() {
		
		final int RUNDA=3;
		
		System.out.println("\n\n==Test regula odpowiedzialnych za dobijanie runda 3==");
		System.out.print("sprawdzenie ");	
		
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
		SWIGTYPE_p_DobijanieRX  ile_grac = ai_texas_swig.getDobijanieRXPTRZReguly(reguly, 3);
		
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
			
			boolean wynik_java = gracz_ai_java.rundaX_dobijanie(stawka_dummy_proponowana, RUNDA);			
			
			
			ai_texas_swig.aplikujDobijanieRXHOST(ile_grac, gra, ktory_gracz, wyniki, stawka_dummy_proponowana);
			

			assertEquals(wynik_java, wyniki[0]==1.0f);
			
			if (i%5000==0)
				System.out.print("... "+i);
			
			gneratorRozdan.generate();
			ai_texas_swig.destruktorGra(gra);
		}
		
		System.out.print("\nTest zakonczony sukcesem");
	}		
	
	
	
	public void testRegulRunda4() {
		
		final int RUNDA=4;
		
		System.out.println("\n\n==Test regula odpowiedzialnych za dobijanie runda 4==");
		System.out.print("sprawdzenie ");	
		
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
		SWIGTYPE_p_DobijanieRX  ile_grac = ai_texas_swig.getDobijanieRXPTRZReguly(reguly, 4);		
		
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
			
			boolean wynik_java = gracz_ai_java.rundaX_dobijanie(stawka_dummy_proponowana, RUNDA);			
			
			
			ai_texas_swig.aplikujDobijanieRXHOST(ile_grac, gra, ktory_gracz, wyniki, stawka_dummy_proponowana);
			

			assertEquals(wynik_java, wyniki[0]==1.0f);
			
			if (i%5000==0)
				System.out.print("... "+i);
			
			gneratorRozdan.generate();
			ai_texas_swig.destruktorGra(gra);
		}
		
		System.out.print("\nTest zakonczony sukcesem");
	}		
	
	
}
