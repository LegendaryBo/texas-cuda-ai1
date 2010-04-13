package pl.wroc.uni.ii.evolution.distribution.eval;

import java.util.ArrayList;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunication;
import pl.wroc.uni.ii.evolution.servlets.databasesupport.communication.EvDBServletCommunicationImpl;
import pl.wroc.uni.ii.evolution.servlets.masterslave.communication.EvMasterSlaveCommunication;
import pl.wroc.uni.ii.evolution.servlets.masterslave.communication.EvMasterSlaveCommunicationImpl;

/**
 * Class implementing operator that evaluates given population using wEvo
 * distribution system.<BR>
 * Population is divided into groups by the main server, which are send to other
 * computers, evaluated there and send back to the main evaluator.<BR>
 * NOTE: The operator must be run from EvMasterSlaveThread (see manual how to do
 * this).
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 * @author Kacper Gorski (admin@34all.org)
 */
public class EvExternalEvaluationOperator<T extends EvIndividual> implements
    EvOperator<T> {

  // delay beetwen checkouts in milicenods
  private long delay;

  // database servlet
  private EvDBServletCommunication db_comm;

  // master slave manager servlet
  private EvMasterSlaveCommunication ev_comm;

  private int task_id = 0;

  private long node_id = 0L;

  private int iteration = 0;

  private boolean fun_sent = false;


  /**
   * @param delay in milliseconds between checkouts if evaluation task is
   *        finished by other computers
   * @param db_comm - database interface
   * @param ev_comm - master slave manager interface]
   * @param wevo_server_url - wEvo main server URL.
   */
  public EvExternalEvaluationOperator(long delay, int task_id, long node_id,
      String wevo_server_url) {
    this.delay = delay;
    this.db_comm = new EvDBServletCommunicationImpl(wevo_server_url);
    this.ev_comm = new EvMasterSlaveCommunicationImpl(wevo_server_url);
    this.node_id = node_id;
    this.task_id = task_id;

  }


  public EvPopulation<T> apply(EvPopulation<T> population) {

    iteration++;

    // Get indexes of individuals whose objective function wasn't evaluated yet
    ArrayList<Integer> to_send = new ArrayList<Integer>();

    for (int i = 0; i < population.size(); i++) {

      if (!population.get(i).isObjectiveFunctionValueCalculated()) {
        to_send.add(i);
      }
    }

    if (to_send.size() == 0) {
      return population;
    }

    // Get table of individual objects to be send
    Object[] pop_to_send = new Object[to_send.size()];
    for (int i = 0; i < to_send.size(); i++) {
      pop_to_send[i] = (population.get(to_send.get(i)));
    }

    try {

      // send objective function to database if not sent yet
      if (!fun_sent && !db_comm.presentFun(task_id)) {
        fun_sent = true;
        db_comm.addFun(task_id, population.get(0).getObjectiveFunction());
      }

      // Send individuals to database and receive its unique ids
      int[] ids =
          db_comm.addIndividualsToEval(task_id, 0, node_id, iteration,
              pop_to_send);

      long work_id = ev_comm.addWork(task_id, ids);

      // regulary checkouts whether coputation is completed

      while (!ev_comm.isWorkDone(work_id)) {
        Thread.sleep(delay);
      }

      // get evaluated values
      double[] values = null;

      values = db_comm.getValues(ids);

      if (values.length != pop_to_send.length) {
        throw new Exception("Corructed values");
      }

      // apply objective function values to population
      int i = 0;
      for (Integer index : to_send) {
        // setting objective function values without evaluating obj_function
        // directly
        population.get(index).assignObjectiveFunctionValue(values[i++]);
      }

    } catch (Exception e) {
      e.printStackTrace(System.out);
    }

    return population;
  }

}
