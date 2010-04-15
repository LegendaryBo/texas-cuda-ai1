package cuda.test;

import engine.Gra;
import engine.TexasSettings;
import generator.GeneratorGraczyZGeneracji;
import generator.GeneratorRozdan;
import generator.IndividualGenerator;

import java.util.Date;
import java.util.Random;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.RegulaAbstrakcyjnaIleGrac;
import wevo.TexasObjectiveFunction;
import Gracze.GraczAIv3;
import Gracze.gracz_v3.GeneratorRegulv3;
import cuda.swig.SWIGTYPE_p_CzyGracR1;
import cuda.swig.SWIGTYPE_p_CzyGracRX;
import cuda.swig.SWIGTYPE_p_DobijanieR1;
import cuda.swig.SWIGTYPE_p_DobijanieRX;
import cuda.swig.SWIGTYPE_p_Gra;
import cuda.swig.SWIGTYPE_p_IleGracR1;
import cuda.swig.SWIGTYPE_p_IleGracRX;
import cuda.swig.SWIGTYPE_p_KodGraya;
import cuda.swig.SWIGTYPE_p_Reguly;
import cuda.swig.SWIGTYPE_p_StawkaR1;
import cuda.swig.SWIGTYPE_p_StawkaRX;
import cuda.swig.SWIGTYPE_p_int;
import cuda.swig.SWIGTYPE_p_p_int;
import cuda.swig.ai_texas_swig;
import cuda.swig.texas_swig;

public class TestFunkcjiCelu extends TestCase {

	final int ILOSC_SPRAWDZEN = 20000;

	public void setUp() {
		TexasSettings.setTexasLibraryPath();
	}

	public static int ile_porownan = 0;

	final int DLUGOSC_KODU = 5;
	final int DLUGOSC_KODU2 = 5;

	/**
	 * Test losuje pewna ilosc rozdan i sprawdza, czy program w C i javie
	 * obliczyly tak samo
	 */
	public void testRunda1() {

		SWIGTYPE_p_Reguly regula = ai_texas_swig.getReguly();
		SWIGTYPE_p_CzyGracR1 czy_grac_c = ai_texas_swig.getCzyGracR1PTRZReguly(regula);
		SWIGTYPE_p_DobijanieR1 dobijanie_c = ai_texas_swig.getDobijanieR1PTRZReguly(regula);
		SWIGTYPE_p_IleGracR1 ile_grac_c = ai_texas_swig.getIleGracR1PTRZReguly(regula);
		SWIGTYPE_p_StawkaR1 stawka_c = ai_texas_swig.getStawkaR1PTRZReguly(regula);
		
		GeneratorRegulv3.init();
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		Random random = new Random(1235);

		IndividualGenerator pIndividualGenerator = new IndividualGenerator(
				12, 2000);
		EvBinaryVectorIndividual[] pIndividual = new EvBinaryVectorIndividual[6];

		for (int j = 0; j < 6; j++)
			pIndividual[j] = pIndividualGenerator.generate();

		SWIGTYPE_p_int osobnik1 = ai_texas_swig.getOsobnikPTR(
				pIndividual[0].getGenes(), 2000);
		SWIGTYPE_p_int osobnik2 = ai_texas_swig.getOsobnikPTR(
				pIndividual[1].getGenes(), 2000);
		SWIGTYPE_p_int osobnik3 = ai_texas_swig.getOsobnikPTR(
				pIndividual[2].getGenes(), 2000);
		SWIGTYPE_p_int osobnik4 = ai_texas_swig.getOsobnikPTR(
				pIndividual[3].getGenes(), 2000);
		SWIGTYPE_p_int osobnik5 = ai_texas_swig.getOsobnikPTR(
				pIndividual[4].getGenes(), 2000);
		SWIGTYPE_p_int osobnik6 = ai_texas_swig.getOsobnikPTR(
				pIndividual[5].getGenes(), 2000);

		System.out.println("\n\n==Test AI rundy 1 ("+ILOSC_SPRAWDZEN+ " gier) ==");
		System.out.println("liczba sprawdzeń... ");
		
		for (int i = 0; i < ILOSC_SPRAWDZEN; i++) {
			int seed = gneratorRozdan.getSeed();

			SWIGTYPE_p_Gra gra = ai_texas_swig.getGraPtr();
				texas_swig.nowaGra(osobnik1,
					osobnik2, osobnik3, osobnik4, osobnik5,
					osobnik6, seed, 2, gra);
			int ktory_gracz = random.nextInt(6);


			Gra gra_java = new Gra(gneratorRozdan);
			float[] wyniki_c = new float[1];
			
			ai_texas_swig.grajRunde1HOST((float) gra_java.bids[ktory_gracz], ile_grac_c, stawka_c, dobijanie_c, czy_grac_c, 
					gra, ktory_gracz, wyniki_c);
			
			GraczAIv3 gracz_ai_java = new GraczAIv3( pIndividual[ktory_gracz], ktory_gracz );
			gracz_ai_java.gra = gra_java;
			float wynik_java = (float) gracz_ai_java.play(1, gra_java.bids[ktory_gracz]);

			assertEquals(wynik_java, wyniki_c[0],0.001f);
			
			if (i%5000==0) {
				System.out.print("... "+i);
			}
			
			gneratorRozdan.generate();
			
			if (i%1000==0) {
				for (int j = 0; j < 6; j++)
					pIndividual[j] = pIndividualGenerator.generate();

				osobnik1 = ai_texas_swig.getOsobnikPTR(
						pIndividual[0].getGenes(), 2000);
				osobnik2 = ai_texas_swig.getOsobnikPTR(
						pIndividual[1].getGenes(), 2000);
				osobnik3 = ai_texas_swig.getOsobnikPTR(
						pIndividual[2].getGenes(), 2000);
				osobnik4 = ai_texas_swig.getOsobnikPTR(
						pIndividual[3].getGenes(), 2000);
				osobnik5 = ai_texas_swig.getOsobnikPTR(
						pIndividual[4].getGenes(), 2000);
				osobnik6 = ai_texas_swig.getOsobnikPTR(
						pIndividual[5].getGenes(), 2000);
			}
			ai_texas_swig.destruktorGra(gra);
		}
		System.out.print(" Test zakonczony pomyslnie");
	}
	
	
	public void testRunda2() {

		final int RUNDA=2;
		
		SWIGTYPE_p_Reguly regula = ai_texas_swig.getReguly();
		SWIGTYPE_p_CzyGracRX czy_grac_c = ai_texas_swig.getCzyGracRXPTRZReguly(regula, 2);
		SWIGTYPE_p_DobijanieRX dobijanie_c = ai_texas_swig.getDobijanieRXPTRZReguly(regula, 2);
		SWIGTYPE_p_IleGracRX ile_grac_c = ai_texas_swig.getIleGracRXPTRZReguly(regula, 2);
		SWIGTYPE_p_StawkaRX stawka_c = ai_texas_swig.getStawkaRXPTRZReguly(regula, 2);
		
		
		System.out.println("\n\n==Test AI rundy 2 ("+ILOSC_SPRAWDZEN+ " gier) ==");
		System.out.println("liczba sprawdzeń... ");
		
		GeneratorRegulv3.init();
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		Random random = new Random(1235);

		IndividualGenerator pIndividualGenerator = new IndividualGenerator(
				12, 2000);
		EvBinaryVectorIndividual[] pIndividual = new EvBinaryVectorIndividual[6];

		for (int j = 0; j < 6; j++)
			pIndividual[j] = pIndividualGenerator.generate();

		SWIGTYPE_p_int osobnik1 = ai_texas_swig.getOsobnikPTR(
				pIndividual[0].getGenes(), 2000);
		SWIGTYPE_p_int osobnik2 = ai_texas_swig.getOsobnikPTR(
				pIndividual[1].getGenes(), 2000);
		SWIGTYPE_p_int osobnik3 = ai_texas_swig.getOsobnikPTR(
				pIndividual[2].getGenes(), 2000);
		SWIGTYPE_p_int osobnik4 = ai_texas_swig.getOsobnikPTR(
				pIndividual[3].getGenes(), 2000);
		SWIGTYPE_p_int osobnik5 = ai_texas_swig.getOsobnikPTR(
				pIndividual[4].getGenes(), 2000);
		SWIGTYPE_p_int osobnik6 = ai_texas_swig.getOsobnikPTR(
				pIndividual[5].getGenes(), 2000);

		for (int i = 0; i < ILOSC_SPRAWDZEN; i++) {
			int seed = gneratorRozdan.getSeed();

			SWIGTYPE_p_Gra gra = ai_texas_swig.getGraPtr();
			texas_swig.nowaGra(osobnik1,
					osobnik2, osobnik3, osobnik4, osobnik5,
					osobnik6, seed, 2, gra);
	
			int ktory_gracz = random.nextInt(6);

			RegulaAbstrakcyjnaIleGrac regula_java;	
			

			
			Gra gra_java = new Gra(gneratorRozdan);
			gra_java.runda=RUNDA-1;
			float[] wyniki_c = new float[1];
			ai_texas_swig.setRunda(gra, RUNDA);
			ai_texas_swig.grajRundeXHOST((float) gra_java.bids[ktory_gracz], ile_grac_c, stawka_c, dobijanie_c, czy_grac_c, 
					gra, ktory_gracz, wyniki_c);
			
			GraczAIv3 gracz_ai_java = new GraczAIv3( pIndividual[ktory_gracz], ktory_gracz );
			
			gracz_ai_java.gra = gra_java;
			float wynik_java = (float) gracz_ai_java.play(RUNDA, gra_java.bids[ktory_gracz]);
			

			assertEquals(wynik_java, wyniki_c[0],0.01f);
			
			if (i%5000==0) 
				System.out.print("... "+i);
			
			
			gneratorRozdan.generate();
			
			if (i%1000==0) {
				
				ai_texas_swig.destruktorInt(osobnik1);
				ai_texas_swig.destruktorInt(osobnik2);
				ai_texas_swig.destruktorInt(osobnik3);
				ai_texas_swig.destruktorInt(osobnik4);
				ai_texas_swig.destruktorInt(osobnik5);
				ai_texas_swig.destruktorInt(osobnik6);
				
				for (int j = 0; j < 6; j++)
					pIndividual[j] = pIndividualGenerator.generate();

				osobnik1 = ai_texas_swig.getOsobnikPTR(
						pIndividual[0].getGenes(), 2000);
				osobnik2 = ai_texas_swig.getOsobnikPTR(
						pIndividual[1].getGenes(), 2000);
				osobnik3 = ai_texas_swig.getOsobnikPTR(
						pIndividual[2].getGenes(), 2000);
				osobnik4 = ai_texas_swig.getOsobnikPTR(
						pIndividual[3].getGenes(), 2000);
				osobnik5 = ai_texas_swig.getOsobnikPTR(
						pIndividual[4].getGenes(), 2000);
				osobnik6 = ai_texas_swig.getOsobnikPTR(
						pIndividual[5].getGenes(), 2000);
			}
			ai_texas_swig.destruktorGra(gra);
		}
		System.out.print(" Test zakonczono pomyslnie");
	}
	
	
	public void testRunda3() {

		final int RUNDA=3;

		SWIGTYPE_p_Reguly regula = ai_texas_swig.getReguly();
		SWIGTYPE_p_CzyGracRX czy_grac_c = ai_texas_swig.getCzyGracRXPTRZReguly(regula, 3);
		SWIGTYPE_p_DobijanieRX dobijanie_c = ai_texas_swig.getDobijanieRXPTRZReguly(regula, 3);
		SWIGTYPE_p_IleGracRX ile_grac_c = ai_texas_swig.getIleGracRXPTRZReguly(regula, 3);
		SWIGTYPE_p_StawkaRX stawka_c = ai_texas_swig.getStawkaRXPTRZReguly(regula, 3);
		
		System.out.println("\n\n==Test AI rundy 3 ("+ILOSC_SPRAWDZEN+ " gier) ==");
		System.out.println("liczba sprawdzeń... ");
		
		GeneratorRegulv3.init();
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		Random random = new Random(1235);

		IndividualGenerator pIndividualGenerator = new IndividualGenerator(
				12, 3000);
		EvBinaryVectorIndividual[] pIndividual = new EvBinaryVectorIndividual[6];

		for (int j = 0; j < 6; j++)
			pIndividual[j] = pIndividualGenerator.generate();

		SWIGTYPE_p_int osobnik1 = ai_texas_swig.getOsobnikPTR(
				pIndividual[0].getGenes(), 2000);
		SWIGTYPE_p_int osobnik2 = ai_texas_swig.getOsobnikPTR(
				pIndividual[1].getGenes(), 2000);
		SWIGTYPE_p_int osobnik3 = ai_texas_swig.getOsobnikPTR(
				pIndividual[2].getGenes(), 2000);
		SWIGTYPE_p_int osobnik4 = ai_texas_swig.getOsobnikPTR(
				pIndividual[3].getGenes(), 2000);
		SWIGTYPE_p_int osobnik5 = ai_texas_swig.getOsobnikPTR(
				pIndividual[4].getGenes(), 2000);
		SWIGTYPE_p_int osobnik6 = ai_texas_swig.getOsobnikPTR(
				pIndividual[5].getGenes(), 2000);

		for (int i = 0; i < ILOSC_SPRAWDZEN; i++) {
			int seed = gneratorRozdan.getSeed();

			SWIGTYPE_p_Gra gra = ai_texas_swig.getGraPtr();
			texas_swig.nowaGra(osobnik1,
					osobnik2, osobnik3, osobnik4, osobnik5,
					osobnik6, seed, 2, gra);
	
			int ktory_gracz = random.nextInt(6);

			RegulaAbstrakcyjnaIleGrac regula_java;	
			

			
			Gra gra_java = new Gra(gneratorRozdan);
			gra_java.runda=RUNDA-1;
			float[] wyniki_c = new float[1];
			ai_texas_swig.setRunda(gra, RUNDA);
			ai_texas_swig.grajRundeXHOST((float) gra_java.bids[ktory_gracz], ile_grac_c, stawka_c, dobijanie_c, czy_grac_c, 
					gra, ktory_gracz, wyniki_c);
			
			GraczAIv3 gracz_ai_java = new GraczAIv3( pIndividual[ktory_gracz], ktory_gracz );
			
			gracz_ai_java.gra = gra_java;
			float wynik_java = (float) gracz_ai_java.play(RUNDA, gra_java.bids[ktory_gracz]);
			

			assertEquals(wynik_java, wyniki_c[0],0.01f);
			
			if (i%5000==0) 
				System.out.print("... "+i);
			
			
			gneratorRozdan.generate();
			
			if (i%1000==0) {
				
				ai_texas_swig.destruktorInt(osobnik1);
				ai_texas_swig.destruktorInt(osobnik2);
				ai_texas_swig.destruktorInt(osobnik3);
				ai_texas_swig.destruktorInt(osobnik4);
				ai_texas_swig.destruktorInt(osobnik5);
				ai_texas_swig.destruktorInt(osobnik6);
				
				for (int j = 0; j < 6; j++)
					pIndividual[j] = pIndividualGenerator.generate();

				osobnik1 = ai_texas_swig.getOsobnikPTR(
						pIndividual[0].getGenes(), 2000);
				osobnik2 = ai_texas_swig.getOsobnikPTR(
						pIndividual[1].getGenes(), 2000);
				osobnik3 = ai_texas_swig.getOsobnikPTR(
						pIndividual[2].getGenes(), 2000);
				osobnik4 = ai_texas_swig.getOsobnikPTR(
						pIndividual[3].getGenes(), 2000);
				osobnik5 = ai_texas_swig.getOsobnikPTR(
						pIndividual[4].getGenes(), 2000);
				osobnik6 = ai_texas_swig.getOsobnikPTR(
						pIndividual[5].getGenes(), 2000);
			}
			
			ai_texas_swig.destruktorGra(gra);
		}
		System.out.print(" Test zakonczono pomyslnie");
	}	
	
	
	
	public void testRunda4() {

		System.out.println("\n\n==Test AI rundy 4 ("+ILOSC_SPRAWDZEN+ " gier) ==");
		System.out.println("liczba sprawdzeń... ");
		
		final int RUNDA=4;
		
		SWIGTYPE_p_Reguly regula = ai_texas_swig.getReguly();
		SWIGTYPE_p_CzyGracRX czy_grac_c = ai_texas_swig.getCzyGracRXPTRZReguly(regula, 4);
		SWIGTYPE_p_DobijanieRX dobijanie_c = ai_texas_swig.getDobijanieRXPTRZReguly(regula, 4);
		SWIGTYPE_p_IleGracRX ile_grac_c = ai_texas_swig.getIleGracRXPTRZReguly(regula, 4);
		SWIGTYPE_p_StawkaRX stawka_c = ai_texas_swig.getStawkaRXPTRZReguly(regula, 4);
		
		
		GeneratorRegulv3.init();
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		Random random = new Random(1235);

		IndividualGenerator pIndividualGenerator = new IndividualGenerator(
				12, 3000);
		EvBinaryVectorIndividual[] pIndividual = new EvBinaryVectorIndividual[6];

		for (int j = 0; j < 6; j++)
			pIndividual[j] = pIndividualGenerator.generate();

		SWIGTYPE_p_int osobnik1 = ai_texas_swig.getOsobnikPTR(
				pIndividual[0].getGenes(), 3000);
		SWIGTYPE_p_int osobnik2 = ai_texas_swig.getOsobnikPTR(
				pIndividual[1].getGenes(), 3000);
		SWIGTYPE_p_int osobnik3 = ai_texas_swig.getOsobnikPTR(
				pIndividual[2].getGenes(), 3000);
		SWIGTYPE_p_int osobnik4 = ai_texas_swig.getOsobnikPTR(
				pIndividual[3].getGenes(), 3000);
		SWIGTYPE_p_int osobnik5 = ai_texas_swig.getOsobnikPTR(
				pIndividual[4].getGenes(), 3000);
		SWIGTYPE_p_int osobnik6 = ai_texas_swig.getOsobnikPTR(
				pIndividual[5].getGenes(), 3000);

		float[] wyniki_c = new float[1];
		
		for (int i = 0; i < ILOSC_SPRAWDZEN; i++) {
			int seed = gneratorRozdan.getSeed();

			SWIGTYPE_p_Gra gra = ai_texas_swig.getGraPtr();
			texas_swig.nowaGra(osobnik1,
					osobnik2, osobnik3, osobnik4, osobnik5,
					osobnik6, seed, 2, gra);
	
			int ktory_gracz = random.nextInt(6);
			
			Gra gra_java = new Gra(gneratorRozdan);
			gra_java.runda=RUNDA-1;
			
			ai_texas_swig.setRunda(gra, RUNDA);
			ai_texas_swig.grajRundeXHOST((float) gra_java.bids[ktory_gracz], ile_grac_c, stawka_c, dobijanie_c, czy_grac_c, 
					gra, ktory_gracz, wyniki_c);
			
			GraczAIv3 gracz_ai_java = new GraczAIv3( pIndividual[ktory_gracz], ktory_gracz );
			
			gracz_ai_java.gra = gra_java;
			float wynik_java = (float) gracz_ai_java.play(RUNDA, gra_java.bids[ktory_gracz]);
			

			assertEquals(wynik_java, wyniki_c[0],0.01f);
			
			if (i%5000==0) 
				System.out.print("... "+i);
			
			
			gneratorRozdan.generate();
			
			if (i%1000==0) {
				
				ai_texas_swig.destruktorInt(osobnik1);
				ai_texas_swig.destruktorInt(osobnik2);
				ai_texas_swig.destruktorInt(osobnik3);
				ai_texas_swig.destruktorInt(osobnik4);
				ai_texas_swig.destruktorInt(osobnik5);
				ai_texas_swig.destruktorInt(osobnik6);
				
				for (int j = 0; j < 6; j++)
					pIndividual[j] = pIndividualGenerator.generate();

				osobnik1 = ai_texas_swig.getOsobnikPTR(
						pIndividual[0].getGenes(), 3000);
				osobnik2 = ai_texas_swig.getOsobnikPTR(
						pIndividual[1].getGenes(), 3000);
				osobnik3 = ai_texas_swig.getOsobnikPTR(
						pIndividual[2].getGenes(), 3000);
				osobnik4 = ai_texas_swig.getOsobnikPTR(
						pIndividual[3].getGenes(), 3000);
				osobnik5 = ai_texas_swig.getOsobnikPTR(
						pIndividual[4].getGenes(), 3000);
				osobnik6 = ai_texas_swig.getOsobnikPTR(
						pIndividual[5].getGenes(), 3000);
			}
			
			ai_texas_swig.destruktorGra(gra);
		}
		System.out.print(" Test zakonczono pomyslnie");
	}
	

	public void testPlay() {

		System.out.println("\n\n==Test AI I ("+ILOSC_SPRAWDZEN+ " partii) ==");
		System.out.println("liczba sprawdzeń... ");
		
		SWIGTYPE_p_Reguly reguly = ai_texas_swig.getReguly();
		
		GeneratorRegulv3.init();
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		Random random = new Random(1235);

		IndividualGenerator pIndividualGenerator = new IndividualGenerator(
				12, 3000);
		EvBinaryVectorIndividual[] pIndividual = new EvBinaryVectorIndividual[6];

		for (int j = 0; j < 6; j++)
			pIndividual[j] = pIndividualGenerator.generate();

		SWIGTYPE_p_int osobnik1 = ai_texas_swig.getOsobnikPTR(
				pIndividual[0].getGenes(), 3000);
		SWIGTYPE_p_int osobnik2 = ai_texas_swig.getOsobnikPTR(
				pIndividual[1].getGenes(), 3000);
		SWIGTYPE_p_int osobnik3 = ai_texas_swig.getOsobnikPTR(
				pIndividual[2].getGenes(), 3000);
		SWIGTYPE_p_int osobnik4 = ai_texas_swig.getOsobnikPTR(
				pIndividual[3].getGenes(), 3000);
		SWIGTYPE_p_int osobnik5 = ai_texas_swig.getOsobnikPTR(
				pIndividual[4].getGenes(), 3000);
		SWIGTYPE_p_int osobnik6 = ai_texas_swig.getOsobnikPTR(
				pIndividual[5].getGenes(), 3000);

		for (int i = 0; i < ILOSC_SPRAWDZEN; i++) {
			int seed = gneratorRozdan.getSeed();

			SWIGTYPE_p_Gra gra = ai_texas_swig.getGraPtr();
			texas_swig.nowaGra(osobnik1,
					osobnik2, osobnik3, osobnik4, osobnik5,
					osobnik6, seed, 2, gra);
	
			int ktory_gracz = random.nextInt(6);

			int runda = random.nextInt(4)+1;
			
			Gra gra_java = new Gra(gneratorRozdan);
			gra_java.runda=runda-1;
			float[] wyniki_c = new float[1];
			ai_texas_swig.setRunda(gra, runda);
			ai_texas_swig.setMode(gra, 3);

			ai_texas_swig.grajHOST(gra, ktory_gracz, wyniki_c,reguly);

			GraczAIv3 gracz_ai_java = new GraczAIv3( pIndividual[ktory_gracz], ktory_gracz );
			
			gracz_ai_java.gra = gra_java;
			float wynik_java = (float) gracz_ai_java.play(runda, gra_java.bids[ktory_gracz]);

		
			assertEquals(wynik_java, wyniki_c[0],0.01f);
			
			if (i%5000==0) 
				System.out.print("... "+i);
			
			
			gneratorRozdan.generate();
			
			if (i%1000==0) {
				
				ai_texas_swig.destruktorInt(osobnik1);
				ai_texas_swig.destruktorInt(osobnik2);
				ai_texas_swig.destruktorInt(osobnik3);
				ai_texas_swig.destruktorInt(osobnik4);
				ai_texas_swig.destruktorInt(osobnik5);
				ai_texas_swig.destruktorInt(osobnik6);
				
				for (int j = 0; j < 6; j++)
					pIndividual[j] = pIndividualGenerator.generate();

				osobnik1 = ai_texas_swig.getOsobnikPTR(
						pIndividual[0].getGenes(), 3000);
				osobnik2 = ai_texas_swig.getOsobnikPTR(
						pIndividual[1].getGenes(), 3000);
				osobnik3 = ai_texas_swig.getOsobnikPTR(
						pIndividual[2].getGenes(), 3000);
				osobnik4 = ai_texas_swig.getOsobnikPTR(
						pIndividual[3].getGenes(), 3000);
				osobnik5 = ai_texas_swig.getOsobnikPTR(
						pIndividual[4].getGenes(), 3000);
				osobnik6 = ai_texas_swig.getOsobnikPTR(
						pIndividual[5].getGenes(), 3000);
			}
			
			ai_texas_swig.destruktorGra(gra);
		}
		System.out.println(" Test zakonczono pomyslnie");		
	}		
		
	
	
	public void testWszystkieRundy() {

	long czas_c=0;
	long czas_java=0;
	long czas_swiga=0;
		
	SWIGTYPE_p_Reguly reguly = ai_texas_swig.getReguly();
	
	System.out.println("\n\n==Test AI II ("+ILOSC_SPRAWDZEN+ " partii) ==");
	System.out.println("liczba sprawdzeń... ");
	
			GeneratorRegulv3.init();
			GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
			Random random = new Random(1235);

			IndividualGenerator pIndividualGenerator = new IndividualGenerator(
					12, 3000);
			EvBinaryVectorIndividual[] pIndividual = new EvBinaryVectorIndividual[6];

			for (int j = 0; j < 6; j++)
				pIndividual[j] = pIndividualGenerator.generate();

			czas_swiga  -= System.nanoTime();
			SWIGTYPE_p_int osobnik1 = ai_texas_swig.getOsobnikPTR(
					pIndividual[0].getGenes(), 3000);
			SWIGTYPE_p_int osobnik2 = ai_texas_swig.getOsobnikPTR(
					pIndividual[1].getGenes(), 3000);
			SWIGTYPE_p_int osobnik3 = ai_texas_swig.getOsobnikPTR(
					pIndividual[2].getGenes(), 3000);
			SWIGTYPE_p_int osobnik4 = ai_texas_swig.getOsobnikPTR(
					pIndividual[3].getGenes(), 3000);
			SWIGTYPE_p_int osobnik5 = ai_texas_swig.getOsobnikPTR(
					pIndividual[4].getGenes(), 3000);
			SWIGTYPE_p_int osobnik6 = ai_texas_swig.getOsobnikPTR(
					pIndividual[5].getGenes(), 3000);
			czas_swiga  += System.nanoTime();

			for (int i = 0; i < ILOSC_SPRAWDZEN; i++) {
				int seed = gneratorRozdan.getSeed();

				czas_swiga  -= System.nanoTime();
				SWIGTYPE_p_Gra gra = ai_texas_swig.getGraPtr();
				texas_swig.nowaGra(osobnik1,
						osobnik2, osobnik3, osobnik4, osobnik5,
						osobnik6, seed, 2, gra);
				czas_swiga  += System.nanoTime();

				Gra gra_java = new Gra( new GraczAIv3[] { 
						new GraczAIv3(pIndividual[0],0),
						new GraczAIv3(pIndividual[1],1),
						new GraczAIv3(pIndividual[2],2),
						new GraczAIv3(pIndividual[3],3),
						new GraczAIv3(pIndividual[4],4),
						new GraczAIv3(pIndividual[5],5)
				} );
				gra_java.rozdanie = gneratorRozdan;

				ai_texas_swig.setMode(gra, 3);
				
				
				czas_c  -= System.nanoTime();
				ai_texas_swig.rozegrajPartieHOST(gra, 2, reguly);

				czas_c  += System.nanoTime();
				czas_java -= System.nanoTime();
				gra_java.play_round(false);
				czas_java += System.nanoTime();
				
				float[] bilans_c = new float[6];
				ai_texas_swig.getBilans(gra, bilans_c);
				

				for (int k=0; k < 6; k++) {
//					System.out.println("gracz "+(k+1)+" "+gra_java.gracze[k].bilans +" "+bilans_c[k]);
					assertEquals(gra_java.gracze[k].bilans, bilans_c[k],0.01f);
				}
//				System.out.println("bla "+(i+1));

//				if (i==1)
//					fail();
				
//				System.out.println(wynik_java);
//				System.out.println(runda);

				
				if (i%5000==0) 
					System.out.print("... "+i);
					
				
				
				gneratorRozdan.generate();
				
				if (i%1000==0) {
					ai_texas_swig.destruktorInt(osobnik1);
					ai_texas_swig.destruktorInt(osobnik2);
					ai_texas_swig.destruktorInt(osobnik3);
					ai_texas_swig.destruktorInt(osobnik4);
					ai_texas_swig.destruktorInt(osobnik5);
					ai_texas_swig.destruktorInt(osobnik6);
					
					for (int j = 0; j < 6; j++)
						pIndividual[j] = pIndividualGenerator.generate();
					czas_swiga  -= System.nanoTime();
					osobnik1 = ai_texas_swig.getOsobnikPTR(
							pIndividual[0].getGenes(), 3000);
					osobnik2 = ai_texas_swig.getOsobnikPTR(
							pIndividual[1].getGenes(), 3000);
					osobnik3 = ai_texas_swig.getOsobnikPTR(
							pIndividual[2].getGenes(), 3000);
					osobnik4 = ai_texas_swig.getOsobnikPTR(
							pIndividual[3].getGenes(), 3000);
					osobnik5 = ai_texas_swig.getOsobnikPTR(
							pIndividual[4].getGenes(), 3000);
					osobnik6 = ai_texas_swig.getOsobnikPTR(
							pIndividual[5].getGenes(), 3000);
					czas_swiga  += System.nanoTime();
				}
				
				gneratorRozdan.generate();
				ai_texas_swig.destruktorGra(gra);
			}
			
			System.out.print(" Test zakonczono pomyslnie");
//			System.out.println("czas java: "+czas_java/1000000.0f+"s");
//			System.out.println("czas c: "+czas_c/1000000.0f+"s");

		}	
	
	
	
	
	
	
}
