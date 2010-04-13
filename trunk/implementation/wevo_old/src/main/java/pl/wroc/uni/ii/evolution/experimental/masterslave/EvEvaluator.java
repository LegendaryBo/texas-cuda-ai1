package pl.wroc.uni.ii.evolution.experimental.masterslave;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import org.mortbay.jetty.HttpConnection;
import org.mortbay.jetty.Request;
import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Evaluates population (computes objective for every individual).
 * 
 * @author Karol 'Asgaroth' Stosiek (karol.stosiek@gmail.com)
 * @author Mateusz 'm4linka' Malinowski (m4linka@gmail.com)
 * @param <T> - subtype of EvIndividual
 */
class EvEvaluator <T extends EvIndividual> {
  
//  private Server jetty;
  
  /**
   * Used logger.
   */
  private final Logger logger = 
      Logger.getLogger(EvEvaluator.class.getCanonicalName());
  
  /**
   * Master slave servlet object.
   */
  private EvMasterServlet master_servlet;
  
  /**
   * Population distributor object.
   */
  private EvPopulationDistributor<T> population_distributor;
  
  /**
   * Master slave servlet class.
   */
  private class EvMasterServlet extends HttpServlet {
  
    /**
     * TODO: read about it and then implement it.
     */
//    private Lock lock;
    
//    private boolean is_evaluating;
    
    /**
     * CLIENT_ID constant.
     */
    private static final String CLIENT_ID_ATTRIBUTE_NAME =
        "CLIENT ID";

    /**
     * Map from client id to population state.
     * 
     * NOTE: Is it necessary to make map concurent? 
     */
    private ConcurrentMap<EvClientID,  EvPopulationState> 
        subpopulations_allocation;
    
    /**
     * List of living clients (more precisely list of ids).
     */
    private List<EvClientID> clients;
    
    /**
     * Constructor.
     */
    public EvMasterServlet() {
//      is_evaluating = false;
      clients = new ArrayList<EvClientID>();
      subpopulations_allocation = 
          new ConcurrentHashMap<EvClientID, EvPopulationState>();
//      population_distributor = new EvEqualPopulationDistributor();
//      population_distribution = new 
//      lock = new ReentrantLock();
    }
    
    /**
     * Gets population.
     * 
     * @return population
     */
    private EvPopulation getPopulation() {
      // wait or something
      return population_distributor.getPopulation();
    }

    /**
     * Registers client.
     * 
     * @param client_id - client id to register
     * @param request - http request
     * @param response - http response
     */
    private void registerClient(
        final EvClientID client_id, 
        final HttpServletRequest request, 
        final HttpServletResponse response) {
      
      throw new UnsupportedOperationException("Not yet implemented");
    }

    /**
     * Sets population.
     * 
     * @param population - population to set
     */
    private void setPopulation(final EvPopulation population) {
      // block list of clients!
//      int clients_len = clients.size();
      population_distributor.distribute(population, clients);
    }
    
    /**
     * Gets subpopulation from a client.
     * 
     * @param request - request from a client
     * @param response - response from server
     */
    @Override
    protected void doPost(final HttpServletRequest request,
        final HttpServletResponse response) {
      
      logger.info("Servicing the POST request...");
      logger.info("Gets subpopulation from client.");
      
      EvClientID client_id =
          new EvClientID(
            (Integer) request.getAttribute(CLIENT_ID_ATTRIBUTE_NAME));
      
      String contentType = request.getContentType();

      if (contentType == null 
          || !contentType.equals("application/octet-stream")) {

        throw new IllegalArgumentException("Request has invalid "
            + "Content-Type. Only application/octet-stream is "
            + "accepted by this servlet.");
      }
      
      logger.info("Reading request contents ...");
      
      try {
        // deserialise data and get population
        
        // get request content
        ObjectInputStream input_stream = 
            new ObjectInputStream(request.getInputStream());

        // change from the Karol Stosiek code
        // NOTE: to examine
        Request base_request;
        if (request instanceof Request) {
          base_request = (Request) request;
        } else {
          base_request = HttpConnection.getCurrentConnection().getRequest();
        }

        Object content = input_stream.readObject();
        
        // signalise that we completed handling the request
        base_request.setHandled(true);
        
        if (content instanceof EvPopulation) {
          // got the population
          EvPopulation<T> subpopulation =
              (EvPopulation<T>) content;
          
          // close the input
          input_stream.close();
          
          // add population to population manager
          population_distributor.setSubpopulation(subpopulation, client_id);

          // doPost method is completed
          logger.info("Deserialization completed, we got population");
        } else {
          response.setStatus(HttpServletResponse.SC_NO_CONTENT);
          throw new Exception("Invalid content, EvPopulation expected!");
        }
      } catch (Exception ex) {
        response.setStatus(HttpServletResponse.SC_BAD_REQUEST);
        Logger.getLogger(
            EvEvaluator.class.getName()).log(Level.SEVERE, null, ex);
      }
    }
    
    /**
     * Sends subpopulation to a client.
     * 
     * @param request - request from a client
     * @param response - response from server
     */
    @Override
    protected void doGet(final HttpServletRequest request,
        final HttpServletResponse response) {
      
      logger.info("Servicing the GET request ...");
      logger.info("Sends subpopulation to client.");
      
      EvClientID client_id =
          new EvClientID(
            (Integer) request.getAttribute(CLIENT_ID_ATTRIBUTE_NAME));
      
      boolean is_client = clients.contains(client_id);
      
      if (is_client) {
        // serialize and send population to client
        
        // change from the Karol Stosiek code
        // NOTE: to examine
        Request base_request;
        if (request instanceof Request) {
          base_request = (Request) request;
        } else {
          base_request = HttpConnection.getCurrentConnection().getRequest();
        }
        
        // signalise that we completed handling the request
        base_request.setHandled(true);
        
        response.setHeader("pragma", "no-cache");
        response.setContentType("application/octet-stream");
        response.setStatus(HttpServletResponse.SC_OK);
        
        try {
          ObjectOutputStream output_stream =
              new ObjectOutputStream(response.getOutputStream());
          
          // subpopulation to send
          EvPopulation<T> subpopulation =
              population_distributor.getSubpopulation(client_id);
          
          output_stream.writeObject(subpopulation);
          output_stream.close();
          
          logger.info("Serialization completed, we sent subpopulation");
        } catch (Exception ex) {
          logger.warning("Sending population to " + client_id.toString() 
              + " was failed!");
          Logger.getLogger(
            EvEvaluator.class.getName()).log(Level.SEVERE, null, ex);
        }
        
      } else {
        // register client
        registerClient(client_id, request, response);
      }
    }
  }

  /**
   * Constructor.
   * 
   * @param ev_population_distributor - population to distribute
   */
  public EvEvaluator(
      final EvPopulationDistributor<T> ev_population_distributor) {
    this.population_distributor = ev_population_distributor;
  }

  /**
   * Evaluates population (for each individual it computes objective).
   * 
   * @param population - population to evaluate
   * @return evaluated population
   */
  public EvPopulation evaluate(final EvPopulation population) {
    this.master_servlet.setPopulation(population);
    return this.master_servlet.getPopulation();
  }
  
}
