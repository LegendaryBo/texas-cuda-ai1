package pl.wroc.uni.ii.evolution.servlets.databasesupport;

import java.io.IOException;

import org.jmock.Mock;
import org.jmock.MockObjectTestCase;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunicationWithErrorRecovery;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunication;
import pl.wroc.uni.ii.evolution.solutionspaces.EvBinaryVectorSpace;

public class EvDBServletCommunicationWithErrorRecoveryTest extends
    MockObjectTestCase {

  private class OneMax implements EvObjectiveFunction<EvBinaryVectorIndividual> {

    /**
     * 
     */
    private static final long serialVersionUID = -7821283340363819848L;

    public double evaluate(EvBinaryVectorIndividual individual) {
      int result = 0;
      for (int i = 0; i < individual.getDimension(); i++) {
        if (individual.getGene(i) == 1) {
          result += 1;
        }
      }
      return result;
    }

  }
  
  // this warning is removed because it uses contructor designed for test purposes only
  @SuppressWarnings("deprecation") 
  public void testNetworkDown() {
    Mock gateway = mock(EvDBServletCommunication.class);
    gateway.expects(atLeastOnce()).method("getSolutionSpace").will(
        throwException(new IOException()));
    EvDBServletCommunication decorator = new EvDBServletCommunicationWithErrorRecovery(
        (EvDBServletCommunication) gateway.proxy(), 4, 20);

    boolean exception_occurred = false;
    try {
      decorator.getSolutionSpace(1, 1);
    } catch (Exception ex) {
      exception_occurred = true;
    }
    assertEquals(true, exception_occurred);
  }

  // this warning is removed because it uses contructor designed for test purposes only
  @SuppressWarnings("deprecation")
  public void testSuccusOnThirdTry() throws Exception {
    EvSolutionSpace subspace = new EvBinaryVectorSpace(new OneMax(), 2);
    Mock gateway = mock(EvDBServletCommunication.class);
    gateway.expects(atLeastOnce()).method("getSolutionSpace").with(eq(1L),
        eq(1L)).will(
        onConsecutiveCalls(throwException(new IOException()),
            throwException(new IOException()), returnValue(subspace)));
    EvDBServletCommunication decorator = new EvDBServletCommunicationWithErrorRecovery(
        (EvDBServletCommunication) gateway.proxy(), 4, 20);

    assertEquals(subspace, decorator.getSolutionSpace(1, 1));
  }
}
