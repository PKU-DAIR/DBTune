select current_timestamp(6) into @query_start;
set @query_name='17.sql';
SELECT SUM(L_EXTENDEDPRICE) / 7.0 AS AVG_YEARLY FROM LINEITEM, PART WHERE P_PARTKEY = L_PARTKEY AND P_BRAND = 'BRAND#44' AND P_CONTAINER = 'WRAP PKG' AND L_QUANTITY < (SELECT 0.2 * AVG(L_QUANTITY) FROM LINEITEM WHERE L_PARTKEY = P_PARTKEY);
set @query_time_ms= timestampdiff(microsecond, @query_start, current_timestamp(6))/1000;
SELECT @query_name, @query_time_ms;