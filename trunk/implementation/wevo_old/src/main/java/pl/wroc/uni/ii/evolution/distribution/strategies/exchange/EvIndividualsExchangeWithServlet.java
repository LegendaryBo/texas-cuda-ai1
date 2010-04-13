package pl.wroc.uni.ii.evolution.distribution.strategies.exchange;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.*;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.structure.EvIndividualInfo;

/**
 * It exchanges individuals through DBGateway
 * 
 * @author Marcin Golebiowski
 */
public class EvIndividualsExchangeWithServlet<T extends EvIndividual>
    implements EvIndividualsExchanger<T> {

  private EvDBServletCommunication gateway;


  /**
   * @param gateway interface to database, used for fetching and sending
   *        indviduals
   */
  public EvIndividualsExchangeWithServlet(EvDBServletCommunication gateway) {
    this.gateway = gateway;
  }


  /**
   * {@inheritDoc}
   */
  public void exportIndividuals(long task_id, long cell_id, long node_id,
      List<T> individuals) throws IOException {
    Object[] array = individuals.toArray(new Object[individuals.size()]);
    double[] values = new double[array.length];
    for (int i = 0; i < values.length; i++) {
      values[i] = individuals.get(i).getObjectiveFunctionValue();
    }

    gateway.addIndividuals(array, task_id, values, cell_id, node_id);
  }


  /**
   * {@inheritDoc}
   */
  @SuppressWarnings("unchecked")
  public List<T> importIndividuals(long[] cells_id, long task_id, int count)
      throws IOException {
    List<T> result = new ArrayList<T>();

    for (long cell : cells_id) {

      EvIndividualInfo[] individiuals_info =
          gateway.getBestIndividualInfosMatchingCell(task_id, cell, 1, count,
              true);

      for (EvIndividualInfo info : individiuals_info) {
        // System.out.println("Import: " + info.getIndividual() + " from node="
        // + info.getNodeID());
        result.add((T) info.getIndividual());
      }

    }
    return result;
  }

}
