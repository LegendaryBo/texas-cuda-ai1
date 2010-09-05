package cuda.test;

import java.util.Random;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import wevo.CUDATexasObjectiveFunction;
import wevo.TaxasSolutionSpace;
import wevo.TexasObjectiveFunction;
import Gracze.gracz_v3.GeneratorRegulv3;

/**
 * Testy na funkcji celu
 * 
 * @author Kacper Gorski (railman85@gmail.com)
 */
public class TestCudaTexasObjectiveFunction extends TestCase {

	private CUDATexasObjectiveFunction cudaObjFunction = null;
	private TexasObjectiveFunction cpuObjFunction = null;
	
	private final int LICZBA_GIER=20000;
	private final int LICZBA_WATKOW=768;
	
	public void setUp() {
		GeneratorRegulv3.init();
		cudaObjFunction = new CUDATexasObjectiveFunction(11, LICZBA_WATKOW, LICZBA_GIER);
		cpuObjFunction = new TexasObjectiveFunction(LICZBA_GIER);
	}
	
	public void atestSingleTest() {
		EvBinaryVectorIndividual individual = getRandomIndividual();
		individual.setObjectiveFunction(cpuObjFunction);
		double wynik_cpu = individual.getObjectiveFunctionValue();
		individual.setObjectiveFunction(cudaObjFunction);
		double wynik_gpu = individual.getObjectiveFunctionValue();
		System.out.println(" wynik cpu "+wynik_cpu);
		System.out.println(" wynik gpu "+wynik_gpu);
		assertEquals(wynik_cpu, wynik_gpu, 500.0);
	}
	
	public void testMultiTest() {
		final int LICZBA_TESTOW=30;
		
		double[] wyniki_java = new double[LICZBA_TESTOW];
		double[] wyniki_c = new double[LICZBA_TESTOW];
		
		EvBinaryVectorIndividual individual = getRandomIndividualFromFile();
		
		for (int i=0; i < LICZBA_TESTOW; i++) {
			cudaObjFunction.usunOsobnikiTreningoweZPamieci();
			cudaObjFunction = new CUDATexasObjectiveFunction(11, LICZBA_WATKOW, LICZBA_GIER);
			individual.setObjectiveFunction(cpuObjFunction);
			wyniki_java[i] = individual.getObjectiveFunctionValue();
			individual.setObjectiveFunction(cudaObjFunction);
			wyniki_c[i] = individual.getObjectiveFunctionValue();
			System.out.println("\ntest nr "+(i+1));
			System.out.println(" wynik cpu "+wyniki_java[i]);
			System.out.println(" wynik gpu "+wyniki_c[i]);
//			assertEquals(wynik_cpu, wynik_gpu, 500.0);
		}
		System.out.println("srednia java "+DaneStatystyczneUtils.getSredniaWartosc(wyniki_java));
		System.out.println("srednia c "+DaneStatystyczneUtils.getSredniaWartosc(wyniki_c));
		System.out.println("odchylenie java "+DaneStatystyczneUtils.getOdchylenie(wyniki_java));
		System.out.println("odchylenie c "+DaneStatystyczneUtils.getOdchylenie(wyniki_c));
	}
	
	
	public void atestDestruktorow() {
		final int LICZBA_TESTOW=50;
		for (int i=0; i < LICZBA_TESTOW; i++) {
			cudaObjFunction = new CUDATexasObjectiveFunction(11, LICZBA_WATKOW, 1000);
			cudaObjFunction.usunOsobnikiTreningoweZPamieci();
		}
	}
	
	
	private EvBinaryVectorIndividual getRandomIndividual() {
		EvBinaryVectorIndividual individual = new EvBinaryVectorIndividual(GeneratorRegulv3.rozmiarGenomu);
		Random rand = new Random();
		
		for (int i=0; i < GeneratorRegulv3.rozmiarGenomu; i++)
			individual.setGene(i, rand.nextInt(2));
		
		return individual;
	}

	private EvBinaryVectorIndividual getRandomIndividualFromFile() {		
		TaxasSolutionSpace solutionSpace = new TaxasSolutionSpace(null, 1, 7);
		int losowa = new Random(3433).nextInt( solutionSpace.lista.size() );
		System.out.println("osonik: "+losowa);
		return solutionSpace.lista.get(losowa );
	}
	
	
}
