package pl.wroc.uni.ii.evolution.distribution2.operators;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Collection;
import java.util.logging.Logger;
import java.util.logging.Level;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.prototype.EvIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvOperator;

/**
 * Simple distribution operator, receiving fixed amount of individuals over the
 * network to a given server. When applied, the operator will send a request to
 * the remote server and try to receive individuals using HTPP protocol.
 * 
 * @author Karol Stosiek (karol.stosiek@gmail.com)
 * @param <T> EvIndividual class
 */
public class EvSimpleDistributionGetOperator<T extends EvIndividual> implements
    EvOperator<T> {

  /**
   * URL address of the server.
   */
  private final URL url;

  /**
   * Number of individuals requested in each application.
   */
  private final Integer amount;

  /**
   * Used to logging.
   */
  private final Logger logger; 
  
  /**
   * URL field accessor.
   * 
   * @return URL of the server
   */
  public URL getUrl() {
    return url;
  }

  /**
   * Amount field accessor.
   * 
   * @return amount of individuals to get
   */
  public Integer getAmount() {
    return amount;
  }


  /**
   * Simple distribution operator, responsible for requesting a fixed number of
   * individuals from the given server.
   * 
   * @param initUrl - server address.
   * @param initAmount - number of individuals requested with each application
   *        of the operator.
   */
  public EvSimpleDistributionGetOperator(
      final URL initUrl, final int initAmount) {

    this.url = initUrl;

    if (initAmount < 0) {
      throw new IllegalArgumentException(
          "It's impossible to pick less than 0 individuals.");
    }

    this.amount = initAmount;
    
    this.logger = Logger.getLogger(
        EvSimpleDistributionGetOperator.class.getCanonicalName());
  }


  /**
   * Opens a new connection with the resource targeted by the URL, for which the
   * operator was created.
   * 
   * @return connection that is created
   * @throws IOException - thrown, when unable to open connection
   */
  private HttpURLConnection openConnection() throws IOException {
    HttpURLConnection connection =
        (HttpURLConnection) this.url.openConnection();

    this.logger.info("Setting up connection with server...");
    
    connection.setDoInput(true);
    connection.setUseCaches(false);
    connection.setRequestMethod("GET");

    connection.addRequestProperty("amount", this.amount.toString());
    connection.connect(); // ensuring we are connected

    this.logger.info("Connected.");
    
    return connection;
  }


  /**
   * Closes the given connection.
   * 
   * @param connection - connection to close
   */
  private void closeConnection(final HttpURLConnection connection) {

    this.logger.info("Disconnecting...");

    connection.disconnect();

    this.logger.info("Disconnected.");
  }


  /**
   * Reads the request contents, deserializes the collection and returns it.
   * 
   * @param connection - connection with the server.
   * @return collection encoded in the request body.
   * @throws Exception - when the body did not contain a valid collection
   */
  @SuppressWarnings("unchecked")
  private Collection<T> getRequestContents(final HttpURLConnection connection)
      throws Exception {

    if (connection.getResponseCode() != HttpURLConnection.HTTP_OK) {
      this.logger.info("Server responsed with status code " 
          + connection.getResponseCode() + ". Did not receive any individuals");
      
      return null;
    }
    
    this.logger.info("Reading response contents...");

    /*
     * Opening an input stream, deserializing the object, checking, if the
     * received object is a collection of individuals and putting into the
     * population.
     */

    ObjectInputStream input =
        new ObjectInputStream(connection.getInputStream());

    Object contents = input.readObject();
    
    if (!(contents instanceof Collection)) {
      this.logger.log(Level.WARNING, "Invalid response content!");
      throw new Exception("Not a collection received!");
    }
    
    this.logger.info("Done.");
    
    return (Collection) contents;
  }

  /**
   * Overridden method. See pl.wroc.uni.ii.evolution.engine for details.
   * 
   * @param population population to apply the operator to
   * @return new population, enriched with received individuals
   */
  @SuppressWarnings("unchecked")
  public EvPopulation<T> apply(final EvPopulation<T> population) {
    try {
      /*
       * Opening a connection to the server, enabling reading from server and
       * telling the proxy servers not to cache the content.
       */

      HttpURLConnection connection = openConnection();

      population.addAll(getRequestContents(connection));

      closeConnection(connection);

    } catch (Exception e) {
      e.printStackTrace();
    }

    return population;
  }
}
