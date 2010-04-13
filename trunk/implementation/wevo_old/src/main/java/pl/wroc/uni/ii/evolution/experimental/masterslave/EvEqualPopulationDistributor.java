package pl.wroc.uni.ii.evolution.experimental.masterslave;

import java.util.List;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Population distributor interface.
 * 
 * @author Karol 'Asgaroth' Stosiek (karol.stosiek@gmail.com)
 * @author Mateusz 'm4linka' Malinowski (m4linka@gmail.com)
 * @param <T> - subtype of EvIndividual
 */
public class EvEqualPopulationDistributor <T extends EvIndividual> 
    implements EvPopulationDistributor <T> {

  /**
   * {@inheritDoc }
   */
  public void distribute(final EvPopulation<T> population, 
      final List<EvClientID> clients) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  /**
   * {@inheritDoc }
   */
  public EvPopulation getSubpopulation(final EvClientID client_id) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  /**
   * {@inheritDoc }
   */
  public void setSubpopulation(final EvPopulation<T> subpopulation, 
      final EvClientID client_id) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  /**
   * {@inheritDoc }
   */
  public EvPopulation<T> getPopulation() {
    throw new UnsupportedOperationException("Not supported yet.");
  }

}
