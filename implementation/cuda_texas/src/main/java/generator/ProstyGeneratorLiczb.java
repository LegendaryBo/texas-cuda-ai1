package generator;

/**
 * Prosty generator licz, zwraca tylko inty
 * @author railman
 */
public class ProstyGeneratorLiczb extends Object {

	// prosty generator (a*STAN+b) mod m 
	private int a=243812;
	private int b=353542;
	private int m=6592981;
	
	private int seed=0;
	
	public ProstyGeneratorLiczb(int aSeed) {
		seed = aSeed;
	}
	
	public int nextInt(int modulo) {
		seed = (a*seed +b) % m;
		return seed%modulo; 
	}
}
