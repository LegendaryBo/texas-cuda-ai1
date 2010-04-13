package pl.wroc.uni.ii.evolution.engine.individuals;

/**
 * An individual for ES(mi, lambda). It extends RealVectorIndividual. It has one
 * more thing: mutation probability vector.
 * 
 * @author Piotr Baraniak, Tomasz Kozakiewicz
 */
public class EvRealVectorWithProbabilitiesIndividual extends
    EvRealVectorIndividual {

  private static final long serialVersionUID = -1763514934865950544L;

  protected double[] probabilities;


  /**
   * It create individual by putting input into certain places.
   * 
   * @param vector chromosome.
   * @param probabilities mutation probability vector.
   */
  public EvRealVectorWithProbabilitiesIndividual(double[] vector,
      double[] probabilities) {
    super(vector);
    if (vector.length != probabilities.length) {
      throw new IllegalArgumentException(
          "Gens vector and probabilities vector haven't the same length.");
    }
    this.probabilities = probabilities;
  }


  /**
   * Create individual with empty vectors, but with proper length.
   * 
   * @param d chromosome length.
   */
  public EvRealVectorWithProbabilitiesIndividual(int d) {
    super(d);
    this.probabilities = new double[d];
  }


  public double getProbability(int i) {
    return probabilities[i];
  }


  public void setProbability(int i, double probability) {
    this.probabilities[i] = probability;
  }


  /**
   * Overriden clone. It create new individual with clones of data vectors as
   * data. After that clone() put to new individual objective function from old
   * one.
   */
  @Override
  public EvRealVectorWithProbabilitiesIndividual clone() {
    EvRealVectorWithProbabilitiesIndividual v1 =
        new EvRealVectorWithProbabilitiesIndividual(vector.clone(),
            probabilities.clone());
    for (int i = 0; i < this.getObjectiveFunctions().size(); i++) {
      v1.addObjectiveFunction(this.getObjectiveFunction(i));
      if (this.isEvaluated(i)) {
        v1.assignObjectiveFunctionValue(getObjectiveFunctionValue(i), i);
      }
    }
    return v1;
  }
}
