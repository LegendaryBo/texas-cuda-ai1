package cuda.test;

import java.util.Random;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import wevo.CUDATexasObjectiveFunction;
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
	
	public void setUp() {
		GeneratorRegulv3.init();
		cudaObjFunction = new CUDATexasObjectiveFunction(11, 128, 1000);
		cpuObjFunction = new TexasObjectiveFunction(1000);
	}
	
	public void testSingleTest() {
		EvBinaryVectorIndividual individual = getRandomIndividual();
		individual.setObjectiveFunction(cpuObjFunction);
		double wynik_cpu = individual.getObjectiveFunctionValue();
		individual.setObjectiveFunction(cudaObjFunction);
		double wynik_gpu = individual.getObjectiveFunctionValue();
		System.out.println(" wynik cpu "+wynik_cpu);
		System.out.println(" wynik gpu "+wynik_gpu);
		assertEquals(wynik_cpu, wynik_gpu, 1.0);
	}
	
	public void testMultiTest() {
		final int LICZBA_TESTOW=50;
		for (int i=0; i < LICZBA_TESTOW; i++) {
			EvBinaryVectorIndividual individual = getRandomIndividual();
			individual.setObjectiveFunction(cpuObjFunction);
			double wynik_cpu = individual.getObjectiveFunctionValue();
			individual.setObjectiveFunction(cudaObjFunction);
			double wynik_gpu = individual.getObjectiveFunctionValue();
			System.out.println("\ntest nr "+(i+1));
			System.out.println(" wynik cpu "+wynik_cpu);
			System.out.println(" wynik gpu "+wynik_gpu);
			assertEquals(wynik_cpu, wynik_gpu, 50.0);
		}
	}
	
	
	public void testDestruktorow() {
		final int LICZBA_TESTOW=50;
		for (int i=0; i < LICZBA_TESTOW; i++) {
			cudaObjFunction = new CUDATexasObjectiveFunction(11, 128, 1000);
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
}
