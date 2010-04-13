package pl.wroc.uni.ii.evolution.engine.individuals;

/**
 * An individual for ES(mi, lambda, ro, kappa). It extends
 * RealVectorWithProbabilitiesIndividual. It has two more things that it: alphas
 * vector to calculate rotation and age which is used for calculate how many
 * iterations does individual exist.
 * 
 * @author Lukasz Witko, Piotr Baraniak
 */
public class EvMiLambdaRoKappaIndividual extends
    EvRealVectorWithProbabilitiesIndividual {

  private static final long serialVersionUID = -5135769402242601904L;

  protected double[] alpha;

  protected int age;


  /**
   * It create individual by putting input into certain places.
   * 
   * @param vector chromosome.
   * @param probabilities mutation probability vector.
   * @param alpha rotation vector.
   */
  public EvMiLambdaRoKappaIndividual(double[] vector, double[] probabilities,
      double[] alpha) {
    super(vector, probabilities);
    if (((vector.length * (vector.length + 1)) / 2) != alpha.length)
      throw new IllegalArgumentException("Alpha vector has wrong length.");

    this.alpha = alpha;
    age = 0;
  }


  /**
   * Create individual with empty vectors, but with proper length.
   * 
   * @param d chromosome length.
   */
  public EvMiLambdaRoKappaIndividual(int d) {
    super(d);
    this.alpha = new double[d * (d + 1) / 2];
    age = 0;
  }


  public void setAlpha(int i, double value) {
    alpha[i] = value;
  }


  public double getAlpha(int i) {
    return alpha[i];
  }


  public void increaseAge() {
    age++;
  }


  public int getAge() {
    return age;
  }


  /**
   * Overriden clone. It create new individual with clones of data vectors as
   * data. After that clone() put to new individual objective function and age
   * from old one.
   */
  @Override
  public EvMiLambdaRoKappaIndividual clone() {
    EvMiLambdaRoKappaIndividual v1 =
        new EvMiLambdaRoKappaIndividual(vector.clone(), probabilities.clone(),
            alpha.clone());
    v1.age = age;
    for (int i = 0; i < this.getObjectiveFunctions().size(); i++) {
      v1.addObjectiveFunction(this.getObjectiveFunction(i));
      if (this.isEvaluated(i)) {
        v1.assignObjectiveFunctionValue(getObjectiveFunctionValue(i), i);
      }
    }
    return v1;
  }


  public int getAlphaLength() {
    return alpha.length;
  }
}
