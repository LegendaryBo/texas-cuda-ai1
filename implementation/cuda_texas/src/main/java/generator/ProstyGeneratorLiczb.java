package generator;

import java.util.Random;

import Gracze.gracz_v3.GeneratorRegulv3;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;

/**
 * Prosty generator licz, zwraca tylko inty
 * @author railman
 */
public class ProstyGeneratorLiczb extends Object {

	// prosty generator (a*STAN+b) mod m 
	int a=65537;
	int b=257;
//	int c=4312;
//	int d=54321;
	
	private int seed=0;
	
	public ProstyGeneratorLiczb() {
		seed = Math.abs( new Random().nextInt() );
	}
	
	public ProstyGeneratorLiczb(int aSeed) {
		seed = aSeed;
	}
	
	public int nextInt(int modulo) {
		seed = a*seed + b;
		if (seed < 0)
			seed -=seed;
		
		return seed%modulo;
	}
	
	public int nextInt() {
		seed = a*seed + b;
		if (seed < 0)
			seed -=seed;
		return seed;
	}

	public EvBinaryVectorIndividual generateIndividual() {

		final int pNumOfGenes = GeneratorRegulv3.rozmiarGenomu;
		int[] geny = new int[1 + pNumOfGenes / 32];
		for (int i = 0; i < geny.length; i++) {
			geny[i] = nextInt();
		}
		EvBinaryVectorIndividual individual = new EvBinaryVectorIndividual(geny, pNumOfGenes);
		
		return individual;
	}
}
