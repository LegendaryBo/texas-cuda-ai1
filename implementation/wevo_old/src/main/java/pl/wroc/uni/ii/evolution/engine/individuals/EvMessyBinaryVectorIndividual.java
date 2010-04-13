package pl.wroc.uni.ii.evolution.engine.individuals;

import java.util.ArrayList;

/**
 * Class for MessyBinaryVectorIndividual. Represents messy individual that
 * expresses a binary vector, allele values are in {true, false}.
 * 
 * @author Marek Szykula (marek.esz@gmail.com)
 * @author Piotr Staszak (stachhh@gmail.com)
 */

public class EvMessyBinaryVectorIndividual extends EvMessyIndividual<Boolean> {

  private static final long serialVersionUID = -262518078515091478L;


  /**
   * Constructor creates new MessyBinaryVectorIndividual with specified genes
   * and alleles.
   * 
   * @param length length of the represented binary vector
   * @param genes list of genes (positions of alleles in expressed genotype),
   *        the length has to be the same to alleles, note: it is not copied, do
   *        not modify it after evaluation of individual
   * @param alleles list of alleles (values of genes), the length has to be the
   *        same to genes, note: it is not copied, do not modify it after
   *        evaluation of individual
   */
  public EvMessyBinaryVectorIndividual(int length, ArrayList<Integer> genes,
      ArrayList<Boolean> alleles) {
    super(length, genes, alleles);
  }


  @Override
  public EvMessyBinaryVectorIndividual clone() {
    EvMessyBinaryVectorIndividual cloned =
        new EvMessyBinaryVectorIndividual(genotype_length,
            new ArrayList<Integer>(genes), new ArrayList<Boolean>(alleles));

    for (int i = 0; i < this.getObjectiveFunctions().size(); i++) {
      cloned.addObjectiveFunction(this.getObjectiveFunction(i));
      if (this.isEvaluated(i)) {
        cloned.assignObjectiveFunctionValue(getObjectiveFunctionValue(i), i);
      }
    }

    return cloned;
  }

}