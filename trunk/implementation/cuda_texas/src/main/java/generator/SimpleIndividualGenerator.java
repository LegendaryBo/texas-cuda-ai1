package generator;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;

public class SimpleIndividualGenerator extends IndividualGenerator {

	private int licznik=-1;
	private EvBinaryVectorIndividual[] osobniki;
	
	public SimpleIndividualGenerator(int seed, int size_, EvBinaryVectorIndividual[] osobniki) {
		super(seed, size_);
		this.osobniki = osobniki;
	}
	

	@Override
	public EvBinaryVectorIndividual generate() {
		licznik++;
		return osobniki[licznik%osobniki.length];
	}
}
