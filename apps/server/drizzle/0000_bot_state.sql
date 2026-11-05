CREATE TABLE `bot_settings` (
	`id` integer PRIMARY KEY NOT NULL,
	`configuration` text NOT NULL,
	`updated_at` integer NOT NULL,
	CONSTRAINT "bot_settings_singleton" CHECK("bot_settings"."id" = 1),
	CONSTRAINT "bot_settings_configuration_json" CHECK(json_valid("bot_settings"."configuration")),
	CONSTRAINT "bot_settings_updated_at" CHECK("bot_settings"."updated_at" >= 0)
);
--> statement-breakpoint
CREATE TABLE `bot_states` (
	`symbol` text PRIMARY KEY NOT NULL,
	`active` integer NOT NULL,
	`updated_at` integer NOT NULL,
	CONSTRAINT "bot_states_active" CHECK("bot_states"."active" IN (0, 1)),
	CONSTRAINT "bot_states_updated_at" CHECK("bot_states"."updated_at" >= 0)
);
