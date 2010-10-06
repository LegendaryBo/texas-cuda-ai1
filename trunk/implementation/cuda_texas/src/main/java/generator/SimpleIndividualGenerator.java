package generator;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;

public class SimpleIndividualGenerator extends IndividualGenerator {

	private EvBinaryVectorIndividual[] osobniki;
	private GeneratorLiczbLosowychSpodJava generator = new GeneratorLiczbLosowychSpodJava();
	
	public SimpleIndividualGenerator(int seed, int size_, EvBinaryVectorIndividual[] osobniki) {
		super(seed, size_);
		this.osobniki = osobniki;
	}
	

	@Override
	public EvBinaryVectorIndividual generate() {
		int random = generator.nextInt();
//		System.out.println(random%osobniki.length + " "+osobniki.length);
		return osobniki[random%osobniki.length];
	}
}
