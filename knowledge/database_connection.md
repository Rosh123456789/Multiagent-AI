# MySQL connection issue

Payment API cannot connect to MySQL.

Possible causes:
- MySQL container is not running.
- Database credentials are incorrect.
- The application is trying to connect before MySQL is ready.

Recommended actions:
- Check database container status.
- Restart MySQL if it exited unexpectedly.
- Restart payment-api after the database is reachable.
