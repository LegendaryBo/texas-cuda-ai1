package Gracze.gracz_v3;

import java.util.Date;
import java.util.Random;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import wevo.TexasObjectiveFunction;
import cuda.swig.SWIGTYPE_p_int;
import cuda.swig.SWIGTYPE_p_p_int;
import cuda.swig.ai_texas_swig;
import engine.TexasSettings;
import generator.GeneratorGraczyZGeneracji;
import generator.IndividualGenerator;

/**
 * Klasa ma metode main, przy pomocy ktorej mozna uruchomic obliczanie funkcji celu
 * @author railman
 */
public class TestIndividuala {

	final int ILOSC_SPRAWDZEN = 20000;

	public TestIndividuala() {
		TexasSettings.setTexasLibraryPath();
	}

	public static int ile_porownan = 0;

	final int DLUGOSC_KODU = 5;
	final int DLUGOSC_KODU2 = 5;

	public static void main(String[] args) {
		TestIndividuala bla = new TestIndividuala();

		
		if (args.length < 3) {
			System.out.println("Do uruchomienia testu niezbedne sa 3 parametry:");
			System.out.println("1) Liczba obliczen funkcji celu (int)");
			System.out.println("2) Liczba partii na jedna funkcje celu (int)");
			System.out.println("3) Liczba watkow na blok w CUDA (int)");
			System.out.println("4) logi (0 lub 1), argument opcjonalny");
			return;
		}
		
		if (args.length > 3 && Integer.parseInt( args[3] )==1)
		TexasObjectiveFunction.LOGI=true;
			
		bla.LICZBA_PARTII =  Integer.parseInt( args[1] );
		bla.LICZBA_WATKOW_NA_BLOK = Integer.parseInt( args[2] );
		bla.LICZBA_TESTOW = Integer.parseInt(args[0]);
		
		bla.testFunkcjiCelu();
	}
	
	int LICZBA_PARTII=10000;
	int LICZBA_WATKOW_NA_BLOK=8;	
	int LICZBA_TESTOW=5;
	
	@SuppressWarnings("deprecation")
	public void testFunkcjiCelu() {
		
		long czas_java=0;
		long czas_c=0;
		
		System.out.println("\n\n==Test funkcji celu==");
		
		GeneratorRegulv3.init();
		Random random = new Random(465);
		final int LICZBA_GENOW=GeneratorRegulv3.rozmiarGenomu;
		
		final int LICZBA_OSOBNIKOW=100;
		final int LICZBA_INTOW= (GeneratorRegulv3.rozmiarGenomu-1)/32 +1;
		
		GeneratorGraczyZGeneracji generator = new GeneratorGraczyZGeneracji(1234, LICZBA_GENOW, 4, true);
		EvBinaryVectorIndividual[] osobniki_java = new EvBinaryVectorIndividual[LICZBA_OSOBNIKOW];

		int[][] osobniki = new int[LICZBA_OSOBNIKOW][];
		for (int i=0; i < LICZBA_OSOBNIKOW; i++) {
			osobniki_java[i] = generator.lista.get( random.nextInt( generator.lista.size() ) );
			osobniki[i] = osobniki_java[i].getGenes();
		}
		
		TexasObjectiveFunction objective_function = new TexasObjectiveFunction(LICZBA_PARTII, LICZBA_OSOBNIKOW, osobniki_java);
		
		SWIGTYPE_p_p_int osobniki_ptr = ai_texas_swig.getIndividualPTRPTR(LICZBA_OSOBNIKOW+1);
		for (int i=0; i < osobniki.length; i++) {
			SWIGTYPE_p_int osobnik_ptr = ai_texas_swig.getOsobnikPTR(osobniki[i], LICZBA_INTOW);
			ai_texas_swig.setIndividualPTR(osobnik_ptr, osobniki_ptr, i);
		}
		
		IndividualGenerator generator_losowy = new IndividualGenerator(123, LICZBA_GENOW);
		
		for (int i=0; i < LICZBA_TESTOW; i++) {	
			
			EvBinaryVectorIndividual obliczany_osobnik = generator_losowy.generate();
			SWIGTYPE_p_int obliczany_osobnik_ptr = ai_texas_swig.getOsobnikPTR(obliczany_osobnik.getGenes(), LICZBA_INTOW);
			

			ai_texas_swig.setIndividualPTR(obliczany_osobnik_ptr, osobniki_ptr, LICZBA_OSOBNIKOW);
			
			float[] wynik_c = new float[1];
			czas_c -= new Date().getTime();
			ai_texas_swig.rozegrajNGierCUDA(LICZBA_OSOBNIKOW, osobniki_ptr , wynik_c, 
					LICZBA_PARTII, LICZBA_INTOW, LICZBA_WATKOW_NA_BLOK);
			czas_c += new Date().getTime();
			
			obliczany_osobnik.setObjectiveFunction(objective_function);
			
			czas_java -= new Date().getTime();
			double wynik_java = obliczany_osobnik.getObjectiveFunctionValue();
			czas_java += new Date().getTime();
			
//			assertEquals(wynik_java, wynik_c[0], 0.01f);
			System.out.println("wynik dla osobnika nr "+(i+1)+":");
			System.out.println("wynik java "+wynik_java);
			System.out.println("wynik CUDA "+wynik_c[0]);
			System.out.flush();
			
			ai_texas_swig.destruktorInt(obliczany_osobnik_ptr);
			
		}
		
		System.out.println("\nCzas oblizcen funkcji celu:");
		System.out.println("czas c: "+czas_c/1000.0f+"s");
		System.out.println("czas java: "+czas_java/1000.0f+"s");
	}
	
}
