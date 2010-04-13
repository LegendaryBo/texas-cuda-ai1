// off LineLength
package pl.wroc.uni.ii.evolution.sampleimplementation.students.grzegorzlisowski;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;

// on LineLength
/**
 * Implementation of the K-deceptive OneMax Problem.
 * 
 * @author Grzegorz Lisowski
 */
public class EvKDeceptiveOneMax implements
    EvObjectiveFunction<EvBinaryVectorIndividual> {

  /**
   * .
   */
  private static final long serialVersionUID = 123L;

  /**
   * Size of a block.
   */
  private final int blockSize;


  /**
   * Constructor.
   * 
   * @param bs Size of a block
   * @throws Exception Exception
   */
  public EvKDeceptiveOneMax(final int bs) throws Exception {
    if (bs >= 0) {
      this.blockSize = bs;
    } else {
      throw new Exception("Bledny parametr rozmiaru bloku");
    }
  }


  /**
   * Evaluates a individual.
   * 
   * @param individual individual
   * @return value
   */
  public double evaluate(final EvBinaryVectorIndividual individual) {
    double globalResult = 0.0;
    double blockResult = 0.0;
    int blockCounter = 0;
    int numberOfZerosInBlock = 0;

    for (int i = 0; i < individual.getDimension(); i++) {
      blockCounter++;
      if (individual.getGene(i) == 1) {
        blockResult++;
      } else {
        numberOfZerosInBlock++;
      }

      if (blockCounter == blockSize) {
        if (numberOfZerosInBlock == blockSize) {
          blockResult = blockSize + 1;
        }

        globalResult += blockResult;
        blockCounter = 0;
        blockResult = 0.0;
        numberOfZerosInBlock = 0;
      }
    }

    return globalResult;
  }

}
