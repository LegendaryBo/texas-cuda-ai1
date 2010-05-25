package cuda.test;

import engine.Gra;
import engine.RegulyGry;
import engine.TexasSettings;
import engine.rezultaty.Rezultat;
import generator.GeneratorRozdan;
import generator.IndividualGenerator;

import java.util.Random;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import reguly.RegulaAbstrakcyjna;
import reguly.LicytacjaNaWejscie.RegulaBardzoWysokieKarty;
import reguly.LicytacjaNaWejscie.RegulaCzyKolorWRece;
import reguly.LicytacjaNaWejscie.RegulaCzyParaWRece;
import reguly.LicytacjaNaWejscie.RegulaStalaStawka;
import reguly.LicytacjaNaWejscie.RegulaWysokieKarty;
import reguly.LicytacjaPozniej.RegulaJestRezultat;
import reguly.LicytacjaPozniej.RegulaLicytujGdyMalaStawka;
import Gracze.GraczAIv3;
import Gracze.gracz_v3.GeneratorRegulv3;
import cuda.swig.SWIGTYPE_p_Gra;
import cuda.swig.SWIGTYPE_p_KodGraya;
import cuda.swig.SWIGTYPE_p_Reguly;
import cuda.swig.SWIGTYPE_p_StawkaR1;
import cuda.swig.SWIGTYPE_p_StawkaRX;
import cuda.swig.SWIGTYPE_p_int;
import cuda.swig.ai_texas_swig;
import cuda.swig.texas_swig;

public class TestRegulStawka extends TestCase {

	final int ILOSC_SPRAWDZEN = 20000;

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
	
		System.out.println("\n\n==Test regula odpowiedzialnych za stawke==");
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
//			float[] rezultat_c = new float[2];
//			float[] rezultat_java = null;
			Gra gra_java = new Gra(gneratorRozdan);
			RegulaAbstrakcyjna regula_java;
			float[] wynik = new float[1];
			float stawka_losowa = 10+random.nextInt(1000);

			
			SWIGTYPE_p_KodGraya kodGraya = ai_texas_swig.getKodGrayaPTR(index_start+1, 5);
			SWIGTYPE_p_KodGraya kodGraya2 = ai_texas_swig.getKodGrayaPTR(index_start+1+5, 5);
			ai_texas_swig.stawkaParaWRekuR1HOST(gra, ktory_gracz, kodGraya, wynik);
			regula_java = new RegulaCzyParaWRece(index_start,pIndividual[ktory_gracz],5);
			double wynik_java = regula_java.aplikujRegule(gra_java, ktory_gracz, pIndividual[ktory_gracz], null);
			assertEquals((float)wynik_java, wynik[0]);
			
			
			ai_texas_swig.stawkaKolorWRekuR1HOST(gra, ktory_gracz, kodGraya, wynik);
			regula_java = new RegulaCzyKolorWRece(index_start,5);
			wynik_java = regula_java.aplikujRegule(gra_java, ktory_gracz, pIndividual[ktory_gracz], null);
			assertEquals((float)wynik_java, wynik[0]);

			ai_texas_swig.stawkaWysokaKartaWRekuR1HOST(gra, ktory_gracz, kodGraya, wynik);
			regula_java = new RegulaWysokieKarty(index_start,5);
			wynik_java = regula_java.aplikujRegule(gra_java, ktory_gracz, pIndividual[ktory_gracz], null);
			assertEquals((float)wynik_java, wynik[0]);			

			ai_texas_swig.stawkaBardzoWysokaKartaWRekuR1HOST(gra, ktory_gracz, kodGraya, wynik);
			regula_java = new RegulaBardzoWysokieKarty(index_start,5);
			wynik_java = regula_java.aplikujRegule(gra_java, ktory_gracz, pIndividual[ktory_gracz], null);
			assertEquals((float)wynik_java, wynik[0]);
	
			ai_texas_swig.stawkaStalaHOST(gra, ktory_gracz, kodGraya, wynik);
			regula_java = new RegulaStalaStawka(index_start, null, 5);
			wynik_java = regula_java.aplikujRegule(gra_java, ktory_gracz, pIndividual[ktory_gracz], null);
			assertEquals((float)wynik_java, wynik[0]);
			
			gra_java.stawka = stawka_losowa;
			ai_texas_swig.setStawka(gra, (int) stawka_losowa);
			ai_texas_swig.StawkaLicytujGdyMalaHOST(gra, ktory_gracz, kodGraya,kodGraya2, wynik);
			regula_java = new RegulaLicytujGdyMalaStawka(index_start, 5, 5);
			wynik_java = regula_java.aplikujRegule(gra_java, ktory_gracz, pIndividual[ktory_gracz], null);
			assertEquals((float)wynik_java, wynik[0]);
			
			int rezultat = random.nextInt(5)+2;
			ai_texas_swig.stawkaWysokaKartaRXHOST(gra, ktory_gracz, kodGraya, wynik, rezultat);
			regula_java = new RegulaJestRezultat(index_start, DLUGOSC_KODU, rezultat);
			wynik_java = regula_java.aplikujRegule(gra_java, ktory_gracz, pIndividual[ktory_gracz], 
					RegulyGry.najlepsza_karta( gneratorRozdan.getAllCards(ktory_gracz)) );
			assertEquals((float)wynik_java, wynik[0]);
			
			
			
			if (i%5000==0)
				System.out.print("... "+i);
			
			ai_texas_swig.destruktorGra(gra);
			ai_texas_swig.destruktorKodGraya(kodGraya);
			ai_texas_swig.destruktorKodGraya(kodGraya2);
		}
		System.out.print("\nTest zakonczony sukcesem");
	}	
	
	
	
	public void testRegulRunda1() {
		
		System.out.println("\n\n==Test stawka, runda 1==");
		System.out.print("sprawdzenie ");
		
		GeneratorRegulv3.init();
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		Random random = new Random(1235);
		
		IndividualGenerator pIndividualGenerator = new IndividualGenerator(12, 2000);
		EvBinaryVectorIndividual[] pIndividual = new EvBinaryVectorIndividual[6];

		SWIGTYPE_p_int[] osobnikPointers = new SWIGTYPE_p_int[6];
		for (int j=0; j < 6; j++) {
			pIndividual[j] = pIndividualGenerator.generate();		
			osobnikPointers[j] = ai_texas_swig.getOsobnikPTR(pIndividual[j].getGenes(), 2000) ;
		}		
		
		SWIGTYPE_p_Reguly reguly = ai_texas_swig.getReguly();
		SWIGTYPE_p_StawkaR1  ile_grac = ai_texas_swig.getStawkaR1PTRZReguly(reguly);
		
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
			
//			float stawka_dummy  = (random.nextInt(1000)+6.0f)*5.0f;
			
			float[] wyniki = new float[1];
			
			
			ai_texas_swig.aplikujStawkaR1HOST(ile_grac, gra, ktory_gracz, wyniki);
			
			Gra gra_java = new Gra(gneratorRozdan);
			
			GraczAIv3 gracz_ai_java = new GraczAIv3( pIndividual[ktory_gracz], ktory_gracz );
			gracz_ai_java.gra = gra_java;
			double wynik_java = gracz_ai_java.rundaX_stawka(1);
			
			assertEquals(wynik_java, wyniki[0], 0.001);
			
			if (i%5000==0)
				System.out.print("... "+i);
			
			ai_texas_swig.destruktorGra(gra);
		}
		
		System.out.print("\nTest zakonczony sukcesem");
	}	
	
public void testRegulRunda2() {
		
	System.out.println("\n\n==Test stawka, runda 2==");
	System.out.print("sprawdzenie ");
	
	GeneratorRegulv3.init();
	GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
	Random random = new Random(1235);
		
	IndividualGenerator pIndividualGenerator = new IndividualGenerator(12, 2000);
	EvBinaryVectorIndividual[] pIndividual = new EvBinaryVectorIndividual[6];

	SWIGTYPE_p_int[] osobnikPointers = new SWIGTYPE_p_int[6];
	for (int j=0; j < 6; j++) {
		pIndividual[j] = pIndividualGenerator.generate();		
		osobnikPointers[j] = ai_texas_swig.getOsobnikPTR(pIndividual[j].getGenes(), 2000) ;
	}		
		
	SWIGTYPE_p_Reguly reguly = ai_texas_swig.getReguly();
	SWIGTYPE_p_StawkaRX  ile_grac = ai_texas_swig.getStawkaRXPTRZReguly(reguly, 2);
		
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
			ai_texas_swig.setStawka(gra, (int) stawka_dummy);
			
			float[] wyniki = new float[1];
			ai_texas_swig.setRunda(gra, 2);
			Gra gra_java = new Gra(gneratorRozdan);
			GraczAIv3 gracz_ai_java = new GraczAIv3( pIndividual[ktory_gracz], ktory_gracz );
			gra_java.runda=1;
			gra_java.stawka=stawka_dummy;
			gracz_ai_java.rezultat = Rezultat.pobierzPrognoze(gra_java, ktory_gracz);
			gracz_ai_java.gra = gra_java;
			
			double wynik_java = gracz_ai_java.rundaX_stawka(2);			
						
			ai_texas_swig.aplikujStawkaRXHOST(ile_grac, gra, ktory_gracz, wyniki);
			
			assertEquals(wynik_java, wyniki[0], 0.001);
			
			if (i%5000==0)
				System.out.print("... "+i);
			
			gneratorRozdan.generate();
			
			ai_texas_swig.destruktorGra(gra);
		}
		
		System.out.print("\nTest zakonczony sukcesem");
	}		
	


public void testRegulRunda3() {
	
	System.out.println("\n\n==Test stawka, runda 3==");
	System.out.print("sprawdzenie ");
	
	GeneratorRegulv3.init();
	GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
	Random random = new Random(1235);
	
	IndividualGenerator pIndividualGenerator = new IndividualGenerator(12, 2000);
	EvBinaryVectorIndividual[] pIndividual = new EvBinaryVectorIndividual[6];

	SWIGTYPE_p_int[] osobnikPointers = new SWIGTYPE_p_int[6];
	for (int j=0; j < 6; j++) {
		pIndividual[j] = pIndividualGenerator.generate();		
		osobnikPointers[j] = ai_texas_swig.getOsobnikPTR(pIndividual[j].getGenes(), 2000) ;
	}	
	
	SWIGTYPE_p_Reguly reguly = ai_texas_swig.getReguly();
	SWIGTYPE_p_StawkaRX  ile_grac = ai_texas_swig.getStawkaRXPTRZReguly(reguly, 3);
	
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
		
//		System.out.println("test nr"+i);

		
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
		ai_texas_swig.setStawka(gra, (int) stawka_dummy);
		
		float[] wyniki = new float[1];
		ai_texas_swig.setRunda(gra, 3);
		Gra gra_java = new Gra(gneratorRozdan);
		GraczAIv3 gracz_ai_java = new GraczAIv3( pIndividual[ktory_gracz], ktory_gracz );
		gra_java.runda=2;
		gra_java.stawka=stawka_dummy;
		gracz_ai_java.rezultat = Rezultat.pobierzPrognoze(gra_java, ktory_gracz);
		gracz_ai_java.gra = gra_java;
		
		double wynik_java = gracz_ai_java.rundaX_stawka(3);			
		
		
		ai_texas_swig.aplikujStawkaRXHOST(ile_grac, gra, ktory_gracz, wyniki);
		
		assertEquals(wynik_java, wyniki[0], 0.001);
		
		if (i%5000==0)
			System.out.print("... "+i);
		
		gneratorRozdan.generate();
		
		ai_texas_swig.destruktorGra(gra);
	}
	System.out.print("\nTest zakonczony sukcesem");
	
}		


	public void testRegulRunda4() {
		
		System.out.println("\n\n==Test stawka, runda 4==");
		System.out.print("sprawdzenie ");
		
		GeneratorRegulv3.init();
		GeneratorRozdan gneratorRozdan = new GeneratorRozdan();
		Random random = new Random(1235);
		
		IndividualGenerator pIndividualGenerator = new IndividualGenerator(12, 2000);
		EvBinaryVectorIndividual[] pIndividual = new EvBinaryVectorIndividual[6];
	
		SWIGTYPE_p_int[] osobnikPointers = new SWIGTYPE_p_int[6];
		for (int j=0; j < 6; j++) {
			pIndividual[j] = pIndividualGenerator.generate();		
			osobnikPointers[j] = ai_texas_swig.getOsobnikPTR(pIndividual[j].getGenes(), 2000) ;
		}	
		
		SWIGTYPE_p_Reguly reguly = ai_texas_swig.getReguly();
		SWIGTYPE_p_StawkaRX  ile_grac = ai_texas_swig.getStawkaRXPTRZReguly(reguly, 4);
		
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
			
	//		System.out.println("test nr"+i);
	
			
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
			ai_texas_swig.setStawka(gra, (int) stawka_dummy);
			
			float[] wyniki = new float[1];
			ai_texas_swig.setRunda(gra, 4);
			Gra gra_java = new Gra(gneratorRozdan);
			GraczAIv3 gracz_ai_java = new GraczAIv3( pIndividual[ktory_gracz], ktory_gracz );
			gra_java.runda=3;
			gra_java.stawka=stawka_dummy;
			gracz_ai_java.rezultat = Rezultat.pobierzPrognoze(gra_java, ktory_gracz);
			gracz_ai_java.gra = gra_java;
			
			double wynik_java = gracz_ai_java.rundaX_stawka(4);			
			
			
			ai_texas_swig.aplikujStawkaRXHOST(ile_grac, gra, ktory_gracz, wyniki);
			
			assertEquals(wynik_java, wyniki[0], 0.001);
			
			if (i%5000==0)
				System.out.print("... "+i);
			
			gneratorRozdan.generate();
			
			ai_texas_swig.destruktorGra(gra);
		}
		
		System.out.print("\nTest zakonczony sukcesem");
	}	

	
	
}
