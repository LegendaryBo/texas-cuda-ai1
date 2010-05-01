package generator;

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
	
	public ProstyGeneratorLiczb(int aSeed) {
		seed = aSeed;
	}
	
	public int nextInt(int modulo) {
		seed = a*seed + b;
		if (seed < 0)
			seed -=seed;
		
		return seed%modulo;
	}
}
