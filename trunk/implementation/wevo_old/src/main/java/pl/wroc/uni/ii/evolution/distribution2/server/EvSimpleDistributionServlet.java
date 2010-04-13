package pl.wroc.uni.ii.evolution.distribution2.server;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.logging.Logger;
import java.util.logging.Level;

import javax.servlet.ServletConfig;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.mortbay.jetty.Request;

import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;

/**
 * Simple container servlet, providing methods to send and receive individuals
 * over the network. Individuals are stored inside the servlet (context?) in a
 * simple, standard java container. This class is under development and though
 * should be not considered as a way of redistributing calculations; use
 * pl.wroc.uni.ii.evolution.servlets.* package instead.
 * 
 * @author Karol Stosiek (karol.stosiek@gmail.com)
 */
public class EvSimpleDistributionServlet extends HttpServlet {

  /**
   * automatically generated to quiet the warnings.
   */
  private static final long serialVersionUID = 514168927218615083L;

  /**
   * Container with received individuals.
   */
  private SortedSet<EvIndividual> individuals;

  /**
   * Used for logging.
   */
  private final Logger logger;


  /**
   * Servlet class, responsible for dealing with HTTP POST and HTTP GET
   * requests. Individuals sent over the network are stored in an inner
   * collection class.
   * 
   * @throws ServletException
   */
  public EvSimpleDistributionServlet() {
    this.logger =
        Logger.getLogger(EvSimpleDistributionServlet.class.getCanonicalName());

    this.individuals =
        Collections.synchronizedSortedSet(new TreeSet<EvIndividual>());
  }


  /**
   * Method responsible for HTTP GET requests. It reads, how many individuals
   * are requested (value of "amount" parameter of the request) and sends as
   * many individuals as possible (but no more than demanded by the client).
   * 
   * @param request - request to serve
   * @param response - response to be returned
   */
  @Override
  protected void doGet(final HttpServletRequest request,
      final HttpServletResponse response) {

    this.logger.info("Servicing a GET request...");

    /*
     * Reading the "amount" parameter; if this parameter is absent in the
     * request, we assume that no individual is requested.
     */
    String amountParameter = request.getHeader("amount");

    int amount = 0;
    if (amountParameter != null) {
      amount = Integer.parseInt(amountParameter);
    }

    this.logger.info("Creating response content...");

    ArrayList<EvIndividual> selected = this.selectIndividuals(amount);

    ((Request) request).setHandled(true);

    response.setHeader("pragma", "no-cache");
    response.setContentType("application/octet-stream");
    response.setStatus(HttpServletResponse.SC_OK);

    try {
      ObjectOutputStream output =
          new ObjectOutputStream(response.getOutputStream());

      output.writeObject(selected);
      output.close();

      this.logger.info("GET request serviced.");

    } catch (IOException e) {
      this.logger.info("Could not write the response content. ");
      this.logger.info("Failed to service GET request.");

      /*
       * Since we have failed to send selected individuals, we have to put them
       * back into container.
       */
      this.individuals.addAll(selected);
    }
  }


  /**
   * Method responsible for handling HTTP POST request. It receives a collection
   * of EvIndividuals and inserts them into the container. Since there is no
   * need for this container to be shared between servlets, it is an element of
   * the servlet instance.
   * 
   * @param request - request to serve
   * @param response - response to be returned
   */
  @Override
  @SuppressWarnings("unchecked")
  protected void doPost(final HttpServletRequest request,
      final HttpServletResponse response) {

    this.logger.info("Servicing a POST request...");
    this.logger.info("Reading request headers...");

    String contentType = request.getContentType();

    if (contentType == null 
        || !contentType.equals("application/octet-stream")) {

      throw new IllegalArgumentException("Request has invalid "
          + "Content-Type. Only application/octet-stream is "
          + "accepted by this servlet.");
    }

    this.logger.info("Reading request contents...");

    try {
      ObjectInputStream input = new ObjectInputStream(request.getInputStream());

      Object contents = input.readObject();
      ((Request) request).setHandled(true);

      if (contents instanceof ArrayList) {
        /*
         * FIXME(karol.stosiek):
         * should I check if the ArrayList contents really are instances of
         * EvIndividual rather than waiting for a potential exception?
         */
        ArrayList<EvIndividual> individual_list =
            (ArrayList<EvIndividual>) contents;

        input.close();

        this.logger.info("Finished reading the request.");

        insertIndividuals(individual_list);

      } else {
        response.setStatus(HttpServletResponse.SC_NO_CONTENT);
        throw new Exception("Invalid content!");
      }

      response.setStatus(HttpServletResponse.SC_ACCEPTED);

      this.logger.info("POST request serviced.");

    } catch (Exception e) {
      response.setStatus(HttpServletResponse.SC_BAD_REQUEST);
      this.logger.log(Level.WARNING, "Failed to serve POST request.");
    }
  }


  /**
   * Method responsible for inserting given individuals into the servlet's
   * container in a synchronized manner. Synchronization is done due to
   * consistency reasons.
   * 
   * @param individual_list - list of individuals to insert
   */
  private synchronized void insertIndividuals(
      final ArrayList<EvIndividual> individual_list) {

    this.individuals.addAll(individual_list);
  }


  /**
   * Method responsible for selecting some number of best individuals out of the
   * servlet's container. This method is synchronized due to consistency
   * reasons.
   * 
   * @param amount - number of individuals to select
   * @return ArrayList of selected individuals
   */
  private synchronized ArrayList<EvIndividual> selectIndividuals(
      final int amount) {

    ArrayList<EvIndividual> selected = new ArrayList<EvIndividual>();

    EvIndividual individual;

    int left = amount;
    while (left > 0) {

      /*
       * If there is no individual left in the container, stop picking
       */
      if (this.individuals.size() == 0) {
        break;
      }

      /*
       * We are guaranteed, that there is at least one individual in the
       * container; remove it from the collection and append to the result
       */
      individual = this.individuals.last();
      this.individuals.remove(individual);
      selected.add(individual);
      left--;
    }

    return selected;
  }


  /**
   * Overriding the servlet's init method.
   * 
   * @param conf - Servlet configuration object.
   * 
   * TODO: avoid overriding.
   */
  @Override
  public void init(final ServletConfig conf) {

  }

  /**
   * Overriding the servlet's init method.
   * TODO: avoid overriding.
   */
  @Override
  public void init() {

  }


  /**
   * Overriding the servlet's init method.
   * TODO: avoid overriding.
   */
  @Override
  public void destroy() {

  }
}
