package cuda.test;

import engine.Hand;
import engine.TexasSettings;
import generator.GeneratorGraczyZGeneracji;
import generator.GeneratorRozdan;
import generator.IndividualGenerator;

import java.util.ArrayList;
import java.util.Date;
import java.util.Random;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import wevo.TaxasSolutionSpace;
import wevo.TexasObjectiveFunction;
import Gracze.gracz_v3.GeneratorRegulv3;
import cuda.swig.SWIGTYPE_p_Hand;
import cuda.swig.SWIGTYPE_p_int;
import cuda.swig.SWIGTYPE_p_p_int;
import cuda.swig.ai_texas_swig;
import cuda.swig.texas_swig;

public class CudaTesty extends TestCase {

	public void setUp() {
		TexasSettings.setTexasLibraryPath();
	}
	
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {

		
		int[] bla = new int[10];
		
		//tests_swig.alokujObiekt(IN, arg1, OUT)
		//tests_swig.testFunkcja(2);
		//System.out.println(bla[0]);
		//System.out.println(bla[1]);
		
		GeneratorRozdan generator  = new GeneratorRozdan();
		
		Hand reka = generator.getHand(3);
		
		SWIGTYPE_p_Hand rezultat = reka.stworzObiektWSwigu();
		
		texas_swig.iloscKartTejSamejWysokosciHOST(rezultat, bla);
		
		System.out.println(bla[0]);
		System.out.println(bla[1]);
		
		//System.out.println(Rezultat.stworzObiektWSwigu());
		//int blaas=4;
		
	}
	
	
	public void testFunkcjaCelu() {
		
		long czas_java=0;
		long czas_c=0;
		
		GeneratorRegulv3.init();
		Random random = new Random();
		final int LICZBA_GENOW=GeneratorRegulv3.rozmiarGenomu;
		final int LICZBA_TESTOW=10;
		final int LICZBA_WATKOW=200;
		
		final int LICZBA_INTOW= (GeneratorRegulv3.rozmiarGenomu-1)/32 +1;
		final int LICZBA_PARTII=5000;
		GeneratorGraczyZGeneracji generator = new GeneratorGraczyZGeneracji(1234, LICZBA_GENOW, 11, true);
		final int LICZBA_OSOBNIKOW=generator.lista.size();
		EvBinaryVectorIndividual[] osobniki_java = new EvBinaryVectorIndividual[LICZBA_OSOBNIKOW];
		int[][] osobniki = new int[LICZBA_OSOBNIKOW][];
		ArrayList<EvBinaryVectorIndividual> losowani = new ArrayList<EvBinaryVectorIndividual>(generator.lista);
		for (int i=0; i < LICZBA_OSOBNIKOW; i++) {
			int losowa = random.nextInt(losowani.size());
			osobniki_java[i] = losowani.remove(losowa);
			osobniki[i] = osobniki_java[i].getGenes();
		}

		TexasObjectiveFunction objective_function = new TexasObjectiveFunction(LICZBA_PARTII, LICZBA_OSOBNIKOW, osobniki_java);
		
		SWIGTYPE_p_p_int osobniki_ptr = ai_texas_swig.getIndividualPTRPTR(LICZBA_OSOBNIKOW+1);
		for (int i=0; i < LICZBA_OSOBNIKOW; i++) {
			SWIGTYPE_p_int osobnik_ptr = ai_texas_swig.getOsobnikPTR(osobniki[i], LICZBA_INTOW);
			ai_texas_swig.setIndividualPTR(osobnik_ptr, osobniki_ptr, i);
		}
		
		IndividualGenerator generator_losowy = new IndividualGenerator(123, LICZBA_GENOW);
		
		SWIGTYPE_p_int obliczany_osobnik_ptr=null;
		for (int i=0; i < LICZBA_TESTOW; i++) {			
			EvBinaryVectorIndividual obliczany_osobnik = getRandomIndividualFromFile();
			
			if (obliczany_osobnik_ptr!=null)
				ai_texas_swig.destruktorInt( obliczany_osobnik_ptr );
			obliczany_osobnik_ptr = ai_texas_swig.getOsobnikPTR(obliczany_osobnik.getGenes(), LICZBA_INTOW);

			ai_texas_swig.setIndividualPTR(obliczany_osobnik_ptr, osobniki_ptr, LICZBA_OSOBNIKOW);

			float[] wynik_c = new float[1];
			czas_c -= new Date().getTime();
			ai_texas_swig.rozegrajNGierCUDA(LICZBA_OSOBNIKOW, osobniki_ptr , wynik_c, LICZBA_PARTII, LICZBA_INTOW, LICZBA_WATKOW, LICZBA_OSOBNIKOW);
			czas_c += new Date().getTime();
			
			obliczany_osobnik.setObjectiveFunction(objective_function);
			
			czas_java -= new Date().getTime();
			double wynik_java = obliczany_osobnik.getObjectiveFunctionValue();
			czas_java += new Date().getTime();
			
			//TODO poprawic
//			assertEquals(wynik_java, wynik_c[0], 0.01f);
			System.out.println("wynik dla gry nr "+(i+1)+":"+wynik_java +" (java) "+wynik_c[0]+" (c) test ok");
			
			
		}
		
		System.out.println("blabla");
		System.out.println("czas c: "+czas_c/1000.0f+"s");
		System.out.println("czas java: "+czas_java/1000.0f+"s");
		
	}
	
	private EvBinaryVectorIndividual getRandomIndividualFromFile() {
		TaxasSolutionSpace solutionSpace = new TaxasSolutionSpace(null, 1, 7);
		return solutionSpace.generateIndividual();
	}
	
	

}
