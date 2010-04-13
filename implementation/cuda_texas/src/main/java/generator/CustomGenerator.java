package generator;

public class CustomGenerator {

	private int seed=0;
	private int prev_seed=0;
	
	public CustomGenerator() {
		
	}
	
	public CustomGenerator(int seed) {
		
		this.seed = seed;
	}
	
	public int nextInt(int ograniczenie) {
		int wylosowana = seed;

		if (wylosowana < 0) 
			wylosowana=-wylosowana;


		seed = (seed * 12991 + 127)%12345789;
		
		return wylosowana%ograniczenie;
	}

	public int getSeed() {
		return seed;
	}
}
